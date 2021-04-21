import os
import re
import subprocess


TOOLS_PATH = '../tools/tools.py'
BART_PATH = os.environ['TOOLBOX_PATH']

ARG_MAP = {}

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
    Get a usage string for a tool
    """
    process = subprocess.Popen(['bart', f'{tool}', '-h'], stdout=subprocess.PIPE)
    help_string = process.communicate()[0].decode('utf-8').split('\n')[0]
    help_string = " ".join(help_string.split()[1:])
    return help_string


def varlen_to_arr(arg_list):
    """
    Convert variable-length array (e.g., dim1 ... dimN) to an array argument
    """
    idx = 0
    for i, arg in enumerate(arg_list):
        if arg == '...':
            idx = i
        if arg[1:] == '[dim':
            arg_list[i][1:] = 'dim'
        if arg[:-1] == 'posn':
            arg_list[i][:-1] = 'pos'
    if idx > 0:
        # check for case arg1 otherarg1 ... argN otherargN
        if arg_list[idx - 2][-1] == '1' and arg_list[idx - 1][-1] == '1':
            first_arg = arg_list[idx - 2][:-1]
            second_arg = arg_list[idx - 1][:-1]
            arg_list = arg_list[:idx - 2] + [f'{first_arg}_arr', f'{second_arg}_arr'] + arg_list[idx + 3:]
        # check for case arg1 arg2 ... argN
        elif arg_list[idx - 1][-1] == '2' and arg_list[idx - 2][-1] == '1':
            if arg_list[idx + 1][-1].lower() == 'n':
                first_arg = arg_list[idx - 1][:-1]
                arg_list = arg_list[:idx - 2] + [f'{first_arg}_arr'] + arg_list[idx + 2:]
        # check for case arg1 ... argN
        elif arg_list[idx - 1][-1] == '1' and arg_list[idx + 1][-1].lower() == 'n':
            first_arg = arg_list[idx-1][:-1]
            arg_list = arg_list[:idx - 1] + [f'{first_arg}_arr'] + arg_list[idx + 2:]
        # casorati case
        else:
            arg_list = ['dim_arr', 'kern_arr', 'input_']
    return arg_list


def get_args_docstring(tool: str):
    """
    Parse function args from help string 

    returns: triplet of (docstring, arglist, kwargs)
    """
    docs_out = subprocess.Popen(['bart', tool, '-h'], stdout=subprocess.PIPE).communicate()
    docs = docs_out[0].decode('utf-8').split('\n')
    # docs[0] contains usage, docs[2] contains description
    doc = docs[0]
    arg_list = []
    args_doc = doc.replace('calibration matrix', 'calibration_matrix')
    args_doc = args_doc.replace('dat file', 'dat_file')
    token_list = re.split(r'\s|\|', args_doc)[2:]
    for token in token_list: # strip the 'Usage: <tool>' tokens
        token = token.strip()
        if token and token[0] != '-':
            if token[0] == '<' or token[-1] == '>':
                token = token.replace('<', '').replace('>', '')
            if len(token) > 1 and token[0] != '[' and token[-1] != ']':
                if token == 'lambda':
                    token = 'llambda'
                    ARG_MAP[token] = 'lambda'
                elif token == 'input':
                    token = 'input_'
                    ARG_MAP[token] = 'input'
                if token != 'output':
                    arg_list.append(token)
    docstring = docs[2]
    flag_idx = 4
    for i, doc in enumerate(docs):
        if len(doc) > 1 and doc[0] == '-':
            flag_idx = i 
            break
    kwarg_list = docs[flag_idx:-1]
    kwarg_dict = {}
    for kwarg in kwarg_list:
        toks = kwarg.split()
        if toks and toks[0][1:]:
            flag = toks[0][1:]
            if flag == '1':
                flag = 'first'
                ARG_MAP['first'] = '1'
            if flag == '3':
                flag = 'third'
                ARG_MAP['third'] = '3'
            split_flags = re.split(r'\s|\/', flag)
            for f in split_flags:
                formatted = f.replace('-', '',)
                kwarg_dict[formatted] = ''
                ARG_MAP[formatted] = f
                if len(toks) > 1:
                    desc = toks[1] + "; " + " ".join(toks[2:])
                    kwarg_dict[formatted] = desc
    return arg_list, docstring, kwarg_dict

def format_docstring(docstring: str, arg_list, kwarg_dict):
    """
    Create docstring / args
    """
    docstring = f'{docstring}\n\n'
    for arg in arg_list:
        docstring += f"\t:param {arg}:\n"
    for kwarg, desc in kwarg_dict.items():
        if 'help' not in kwarg:
            docstring += f"\t:param {kwarg}: {desc}\n"
    return docstring


def create_arg_str(arg_list, kwarg_dict):
    """
    Create argument tuple

    :param arg_list: list of required arguments
    :param kwarg_dict: dictionary of keyword arguments
    """
    arg_str = '('
    format_string = lambda s: "_".join(re.split(r'\s|-|\/', s))
    formatted_args = []
    for arg in arg_list:
        formatted = format_string(arg)
        formatted_args.append(formatted)
        ARG_MAP[formatted] = arg
    arg_list = varlen_to_arr(formatted_args)
    if len(arg_list) > 0:
        arg_str += ', '.join(arg_list) + ', '
    for kwarg, val in kwarg_dict.items():
        # TODO: do this in a cleaner way than the strange indexing
        if 'help' not in val: # don't add help string as an arg
            arg_str += f'{kwarg}=None, '
    arg_str += ')'
    return arg_str


def create_template(tool: str):
    """
    Create 
    """
    arg_list, docstring, kwarg_dict = get_args_docstring(tool)
    formatted_docstring = format_docstring(docstring, arg_list, kwarg_dict)
    arg_str = create_arg_str(arg_list, kwarg_dict)

    tool = tool.decode('utf-8')
    help_string = get_help_string(tool)

    template =f"""
    def {tool}{arg_str}:
    \"\"\"
    {formatted_docstring}
    \"\"\"
    help_string = \"{help_string}\"
    if 'output' in help_string:
        print('output is here')
    print(help_string)
    """.strip()
    return template


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
    

    