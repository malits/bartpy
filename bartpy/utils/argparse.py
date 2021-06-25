import json
import os
import re
import subprocess


TOOLS_PATH = '../tools/tools.py'
BART_PATH = os.environ['TOOLBOX_PATH']

# TODO: better handling for OPT SELECT

ARG_MAP = {}
TYPE_MAP = {
    "INT": 'int',
    "UINT": 'int',
    "FLOAT": 'float',
    "SET": 'bool',
    "SELECT": 'bool',
    "INFILE": 'array',
    "ULONG": 'long'
}

def argparse(**kwargs):
    """
    Parse Pythonic keyword args to command-line flags
    """
    argv = ""
    argc = 1
    for flag, value in kwargs.items():
        argv += f"-{flag} {value} "
        argc += 1
    return argc, argv.rstrip()


def get_tools():
    """
    Gets a list of BART tools that the user has installed
    """
    process = subprocess.Popen([f'{BART_PATH}/bart'], stdout=subprocess.PIPE)
    bart_out = process.communicate()[0].split()
    return bart_out

def get_help_string(tool: str):
    """
    Get command-line help string to use for syscall template
    """
    process = subprocess.Popen([f'{BART_PATH}/bart', tool, '-h'], stdout=subprocess.PIPE)
    bart_out = process.communicate()[0].decode('utf-8')
    usage = bart_out.split("\n")[0]
    usage = re.sub(r'[\<\>]', '', usage)
    return usage.lstrip('Usage: ').strip()


def get_interface_docstring(tool: str):
    """
    Get interface docstring and parse arguments appropriately

    returns: dictionary to create template
    """
    docs_out = subprocess.Popen([f'{BART_PATH}/bart', tool, '--interface'], stdout=subprocess.PIPE).communicate()
    docs = docs_out[0].decode('utf-8')
    docs = re.split(r'name:|usage_str:|help_str:|positional\ arguments:|options:', docs)
    # first element of these arrays is empty str
    name, usage_str, docstring, pos_str, opt_args = docs[1:]
    arg_list = parse_pos_args(pos_str)
    #TODO: remove when tuples are handled
    if not arg_list:
        return
    arg_list = arg_list + parse_opt_args(opt_args)
    pos_args, kw_args = [], []
    has_output = False
    for arg in arg_list:
        if arg['required']:
            pos_args.append(arg)
        elif not arg['required'] and arg['type'] != 'OUTFILE':
            kw_args.append(arg)
        if arg['type'] == 'OUTFILE':
            has_output = True
    template_dict = {
        'name': "_".join(name.split()),
        'usage_str': get_help_string(tool),
        'docstring': docstring,
        'arg_list': pos_args,
        'kwarg_list': kw_args,
        'has_output': has_output
    }
    return template_dict


def parse_pos_args(pos_str: str):
    """
    Convert interface to argument dictionary

    returns: list of dictionaries of the form:
    {
        name: str,
        required: bool,
        input: bool,
        type: type,
        opt: False
    }
    A required argument is a positional argument. A keyword argument is not required.
    """
    arg_list = []
    # pos_str = re.sub(r"\{", r"(", pos_str)
    # pos_str = re.sub(r"\}", r")", pos_str)
    positional_args = pos_str.split("\n")[1:]
    for arg in positional_args:
        if not arg:
            continue
        arg = re.sub(r'[\{\}\"]', '', arg).strip()
        args = arg.split(',')
        required_str, arg_type, num_arg = [x.strip() for x in args[:3]]
        required = False
        if required_str == 'true':
            required = True
        if arg_type == 'ARG_TUPLE':
            print("TODO: Implement tuple handling")
            return
        elif arg_type == 'ARG':
            opt_type, _, arg_name = [x.strip() for x in args[-3:]]
            opt_type = re.sub('OPT_', '', opt_type)
            type_str = opt_type if opt_type not in TYPE_MAP.keys() else TYPE_MAP[opt_type]
            is_input = True
            if opt_type == 'OUTFILE':
                is_input = False
            arg_list.append({
                'name': format_string(arg_name),
                'required': required,
                'input': is_input,
                'type': type_str,
                'opt': False,
            })
    return arg_list

        
def parse_opt_args(opt_str: str):
    """
    Parse optional args

    {
        name: str,
        flag: str,
        required: ,
        type: ,
        opt: True,
    }

    """
    opt_list = []
    opt_str = re.sub(r'[\{\}\"]', '', opt_str)
    opts = opt_str.split("\n")
    for opt in opts:
        if not opt:
            continue
        toks = [tok.strip() for tok in opt.split(',')]
        flag, long_opt, required, opt_type, __ = toks[:5]
        is_long_opt = False
        if long_opt != "(null)":
            flag = long_opt # parse long opt
            is_long_opt = True
        desc = " ".join(toks[5:])
        required = False
        if required == 'true':
            required = True
        type_str = re.sub('OPT_', '', opt_type)
        type_str = type_str if type_str not in TYPE_MAP.keys() else TYPE_MAP[type_str]
        opt_list.append({
            'name': format_string(flag),
            'flag': flag,
            'required': required,
            'type': type_str,
            'opt': True,
            'desc': desc,
            'is_long_opt': is_long_opt
        })
    return opt_list


def format_docstring(usage: str, arg_list, kwarg_dict):
    """
    Create docstring / args
    """
    usage = usage.strip().strip("\"")
    docstring = f'{usage}\n\n'
    for arg in arg_list:
        if not arg['input']:
            continue
        docstring += f"\t:param {arg['name']} {arg['type']}:\n"
    for kwarg in kwarg_dict:
        if 'help' not in kwarg:
            desc = None if 'desc' not in kwarg.keys() else kwarg['desc']
            docstring += f"\t:param {kwarg['name']} {kwarg['type']}: {desc} \n"
    return docstring

def format_string(s):
    """
    Catch-all for formatting strings

    """
    formatted = "_".join(re.split(r'[^_A-Za-z0-9\d]', s))
    # TODO: better way of handling syntactically invalid args
    if re.match(r'\b[0-9]\b', formatted):
        formatted = f'_{formatted}'
    if 'lambda' in formatted:
        formatted = 'llambda'
    return formatted


def create_arg_str(arg_dict, kwarg_dict):
    """
    Create argument tuple

    :param arg_dict: list of required arguments
    :param kwarg_dict: dictionary of keyword arguments
    """
    arg_str = '('
    formatted_args = []
    for arg in arg_dict:
        if arg['type'] == 'OUTFILE':
            continue
        arg_name = arg['name']
        formatted_args.append(arg_name)
        ARG_MAP[arg_name] = arg_name
    #arg_list = varlen_to_arr(formatted_args)
    arg_list = formatted_args
    if len(arg_list) > 0:
        arg_str += ', '.join(arg_list) + ', '
    for kwarg in kwarg_dict:
        name = kwarg['name']
        name = format_string(name)
        if 'help' not in name: # don't add help string as an arg
            arg_str += f'{name}=None, '
    arg_str = arg_str.rstrip(', ')
    arg_str += ')'
    return arg_str


def create_template(tool: str):
    """
    Create template
    """
    template_dict = get_interface_docstring(tool)
    #TODO: remove when tuples are handled
    if not template_dict:
        return ' '
    docstring, arg_list, kwarg_list = \
        template_dict['docstring'], template_dict['arg_list'], template_dict['kwarg_list']
    formatted_docstring = format_docstring(docstring, arg_list, kwarg_list)
    arg_str = create_arg_str(arg_list, kwarg_list)
    
    usage_string = template_dict['usage_str']

    tool = tool.decode('utf8')

    template =f"""
    def {tool}{arg_str}:
    \"\"\"
    {formatted_docstring}
    \"\"\"
    usage_string = \"{usage_string}\"
    """.strip()

    template += '\n'

    # if len(kwarg_list) > 0:
    #     kwarg_list_str = 'flags = ['
    #     for kwarg in kwarg_list:
    #         flag = kwarg['name']
    #         kwarg_list_str += flag + ', '
    #         template += '\n\t' + flag + ' = ' + flag + ' and None'
    #     kwarg_list_str = kwarg_list_str.rstrip(', ') + ']\n'

    #     template += '\n\t' + kwarg_list_str

    # template += f"\n\n\tcmd_str = BART_PATH + '{tool} '\n"

    template += "\n    cmd_str = f'{BART_PATH} '"
    template += f"\n    cmd_str += '{tool} '"

    for kwarg in kwarg_list:
        if kwarg['opt']:
            flag = kwarg['flag']
            arg_name = kwarg['name']
            template += f"\n    if " + arg_name + ":\n        "
            if kwarg['is_long_opt']:
                template += f"cmd_str += f'--{flag} "
            else:
                template += f"cmd_str += f'-{flag} "
            if kwarg['type'] != 'bool':
                template += '{' + arg_name + '} '
            template += "'"

    arg_names = " ".join([arg['name'] for arg in arg_list])
    template += f'\n    cmd_str += \"{arg_names} \"'

    for arg in arg_list:
        name = arg['name']
        if arg['input']:
            template += f"\n    cfl.writecfl(\'{name}\', {name})"

    template += "\n\n    print(cmd_str)\n\n    os.system(cmd_str)\n"

    # TODO: fix optional output (estdelay)
    if template_dict['has_output']:
        return_str = '\n    return '
        for arg in arg_list:
            name = arg['name']
            if not arg['input']:
                return_str += f"cfl.readcfl('{name}'), "
        template += return_str.rstrip(', ')

    return template.strip()


def write_tool_methods():
    """
    Autogenerate `tools.py` file which contains the BART tools
    """
    if not os.path.exists(TOOLS_PATH):
        with open(TOOLS_PATH, 'w+'):
            pass
    template_str = 'from ..utils import cfl\nimport os\n\n\n'
    template_str += "BART_PATH=os.environ['TOOLBOX_PATH'] + '/bart'\n\n\n"
    tool_lst = get_tools()[4:]
    for tool in tool_lst:
        template_str += create_template(tool)
        template_str += '\n\n'
    with open(TOOLS_PATH, 'w+') as f:
        f.write(template_str)

if __name__ == '__main__':
    write_tool_methods()

    