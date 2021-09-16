import json
import os
import re
import subprocess

# NOTE: `MULTITUPLE` refers to an argument of tuples that are passed in to the command line in alternating order
# e.g., dim0 size0 ... dimN sizeN

# Directory for writing the tools file, BART env variable
TOOLS_PATH = 'bartpy/tools/tools.py'
BART_PATH = os.environ['TOOLBOX_PATH']

# map arg names that are reserved in python (e.g., `lambda`) to python-friendly names
# map types from BART interface
ARG_MAP = {}
TYPE_MAP = {
    "INT": 'int',
    "UINT": 'int',
    "FLOAT": 'float',
    "SET": 'bool',
    "SELECT": 'bool',
    "INFILE": 'array',
    "ULONG": 'long',
    "LONG": 'long',
    "TUPLE": 'tuple',
    "ARG_TUPLE": 'tuple',
    "ARG_MULTITUPLE": 'multituple',
    "VEC3": 'list',
    "CLEAR": 'bool'
}

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


def parse_interface(tool: str):
    """
    Use BART's interface option to parse out necessary information for function template
    This is used to generate the respective function in `tools.py`

    this returns a template dictionary containing the following information:
        name: tool name, formatted to be valid in Python
        usage_str: example of how to use the tool on the command line
        docstring: description of the tool's purpose
        arg_list: list of positional (required arguments for which order matters) arguments
        kwarg_list: list of keyword arguments; not always required, one must specify the argument name when calling
        has_output: boolean that is true if the function has an output file

    returns: dictionary to create template
    """
    docs_out = subprocess.Popen([f'{BART_PATH}/bart', tool, '--interface'], stdout=subprocess.PIPE).communicate()
    docs = docs_out[0].decode('utf-8')
    docs = re.split(r'name:|usage_str:|help_str:|positional\ arguments:|options:', docs)

    name, usage_str, docstring, pos_str, opt_args = docs[1:] # first element of these arrays is empty str
    arg_list = parse_pos_args(pos_str) + parse_opt_args(opt_args)
    pos_args, kw_args = [], []
    has_output = False

    for arg in arg_list:
        if arg['required']:
            pos_args.append(arg)
        elif not arg['required'] and arg['type'] != 'OUTFILE':
            kw_args.append(arg)
        elif not arg['required'] and arg['type'] == 'OUTFILE':
            pos_args.append(arg)
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

    returns: list of dictionaries, one for each argument, of the form:
    {
        name: string giving the positional arguments name,
        required: boolean indicating whether or not the argument is required,
        input: boolean indicating whether or not this argument is an input argument (e.g., array) vs output (return value),
        type: type string specifying the type of argument,
        opt: boolean set to true if this is a keyword argument / command line option (always false for positional args)
    }

    A required argument is a positional argument. A keyword argument is not required.
    """
    arg_list = []

    num_tuples = 0
    positional_args = pos_str.split("\n{")[1:]
    for arg in positional_args:
        if not arg: # pass over empty strings
            continue
        
        # Regex for cleaning and parsing interface output
        arg = re.sub(r'[\{\}\"]', '', arg).strip()
        args = arg.split(',')
        if len(args) < 3: # ignore output that does not pertain to an argument
            continue
        
        required_str, arg_type, num_arg = [x.strip() for x in args[:3]]

        required = False       
        if required_str == 'true':
            required = True
        
        # num_arg > 1 indicates a multituple (more than one arg/tuple required)
        if arg_type == 'ARG_TUPLE' and num_arg != '1': 

            num_tuples = int(num_tuples)
            # tuple_args is a list of tuple argument interface strings
            # each element of tuple_arg_lists is a list of the form [OPTION_TYPE, SIZE, NAME]
            tuple_args = [x.strip() for x in arg.split('\n\t')[1:]]
            tuple_arg_lists = [[s.strip() for s in lst.split(",")] for lst in tuple_args]
            arg_list.extend([create_arg_dict(lst, 'ARG_MULTITUPLE', required) for lst in tuple_arg_lists])      
        elif 'ARG' in arg_type:
            # for non-multituple args, last three elements of the 'arg' output are OPTION_TYPE, SIZE, and NAME
            arg_dict = create_arg_dict(args[-3:], arg_type, required)
            arg_list.append(arg_dict)
    
    return arg_list


def create_arg_dict(arg_data, arg_type, required):
    """
    Create dictionary of argument attributes to write the necessary function templates

    Argument dictionary form is repeated here for convienience:
    {
        name: string giving the positional arguments name,
        required: boolean indicating whether or not the argument is required,
        input: boolean indicating whether or not this argument is an input argument (e.g., array),
        type: type string specifying the type of argument,
        opt: boolean set to true if this is a keyword argument / command line option (always false for positional args)
    }
    """

    opt_type, _, arg_name = [x.strip() for x in arg_data]
    opt_type = re.sub('OPT_', '', opt_type)
    type_str = opt_type if opt_type not in TYPE_MAP.keys() else TYPE_MAP[opt_type]
    
    is_input = True
    if opt_type == 'OUTFILE':
        is_input = False

    # Edge cases to handle multituples
    if arg_type == 'ARG_MULTITUPLE':
        type_str = 'multituple'
    elif arg_type == 'ARG_TUPLE':
        type_str = 'tuple'
    
    arg_dict = {
        'name': format_string(arg_name),
        'required': required,
        'input': is_input,
        'type': type_str,
        'opt': False,
    }

    return arg_dict


def parse_opt_args(opt_str: str):
    """
    Parse command-line options (e.g., `-i` to enable inverse FFT) as keyword arguments

    Returns a dictionary where each item takes the form:
    {
        'name': name drawn from cmd line flag (formatted), 
        'flag': flag from cmd line,
        'required': boolean indicating whether or not its required,
        'type': string indicating argument type,
        'opt': boolean indicating whether or not argument is keyword argument / command line option (true for option args),
        'desc': description string,
        'is_long_opt': true if long option,
        'input': true for all keyword arguments (all are inputs),
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

        # handle long options (indicated by `--` on the command line)
        is_long_opt = False
        if long_opt != "(null)": 
            flag = long_opt # parse long opt
            is_long_opt = True
        
        # remaining tokens are description string
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
            'is_long_opt': is_long_opt,
            'input': True,
        })
    
    return opt_list


def format_docstring(usage: str, arg_list, kwarg_dict):
    """
    Use argument lists to create Python docstring
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
    if re.match(r'\b[0-9]\b', formatted):
        formatted = f'_{formatted}'
    if 'lambda' in formatted: # edge case because Python reserves the token `lambda`
        formatted = 'llambda'
    return formatted


def create_arg_str(arg_dict, kwarg_dict):
    """
    Create argument tuple in the function signature

    e.g., for function foo(bar, x, y, z), (bar, x, y, z) is the tuple

    :param arg_dict: list of required arguments
    :param kwarg_dict: dictionary of keyword arguments
    """
    arg_str = '('
    formatted_args = []
    input_args = []

    for arg in arg_dict:
        if arg['type'] == 'OUTFILE': # output file arguments on the command line are return vals in python
            continue
        arg_name = arg['name']
        if arg['input'] and arg['type'] == 'array':
            input_args.append(arg_name)
        else:
            formatted_args.append(arg_name)
            ARG_MAP[arg_name] = arg_name
    
    formatted_args = input_args + formatted_args
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
    Master function to write template string.
    """
    template_dict = parse_interface(tool)
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

    template += "\n\tcmd_str = f'{BART_PATH} '"
    template += f"\n\tcmd_str += '{tool} '"

    arg_names = ""

    template += f"\n\tflag_str = ''\n"
    template += f"\n\topt_args = f''\n"

    has_multituple = False
    template += f"\n\tmultituples = []\n"

    # template += f"\n\tif not os.path.exists('tmp'):"
    # template += f"\n\t\tos.makedirs('tmp')\n"

    for kwarg in kwarg_list:
        if kwarg['opt']:
            flag = kwarg['flag']
            arg_name = kwarg['name']
            if kwarg['type'] == 'array':
                template += f"\n\tif not isinstance({arg_name}, type(None)):\n\t"
                template += f"\tcfl.writecfl(NAME + \'{arg_name}\', {arg_name})\n\t"
            else:
                template += f"\n\tif " + arg_name + " is not None:\n\t"
            if kwarg['is_long_opt']:
                template += f"\tflag_str += f'--{flag} "
            elif kwarg['type'] == 'array':
                template += f"\tflag_str += '-{flag} "
            else:
                template += f"\tflag_str += f'-{flag} "
            
            if kwarg['type'] == 'array':
                template += arg_name + " '\n"
            elif kwarg['type'] == 'list':
                template +=  "{" + f"\":\".join([str(x) for x in {arg_name}])" + "}" + " '\n"
            elif kwarg['type'] == 'bool':
                template += "'\n"    
            else:
                template += '{' + arg_name + "} '\n"

        if not kwarg['opt']:
            name = kwarg['name']
            if kwarg['type'] == 'array':
                template += f"\n\tif not isinstance({name}, type(None)):\n\t"
            else:
                template += f"\n\tif " + name + " != None:\n\t\t"
            if kwarg['type'] == 'tuple':
                template += "opt_args += f\"{" + f"\' \'.join([str(arg) for arg in {name}])" + "} \"\n"
            elif kwarg['type'] == 'multituple':
                template += f"multituples.append({name}) \n"
            elif kwarg['type'] == 'array':
                template += "\topt_args += 'NAME + {" + name + "}'\n" 
            else:
                template += "\topt_args += '{" + name + "}'\n" 
            
    template += "\tcmd_str += flag_str + opt_args + '  '\n"

    arg_names = "{" + f"' '.join([' '.join([str(x) for x in arg]) for arg in zip(*multituples)]).strip()" + "} "

    output_names = ""
    input_names = ""

    for arg in arg_list:
        if arg['type'] == 'array' or arg['type'] == 'OUTFILE':
            arg_names += "{NAME}" + arg['name'] + " "
        elif arg['input'] and arg['type'] == 'tuple':
            arg_names += "{" + f"' '.join([str(arg) for arg in {arg['name']}])" + "} " 
        elif arg['input'] and arg['type'] == 'multituple':
            template += f"\n\tmultituples.append({arg['name']})\n\t"
        else:
            arg_names += "{" + arg['name'] + "} "
    
    template += f'\n\tcmd_str += f\"{arg_names} \"'

    for arg in arg_list:
        name = arg['name']
        if arg['input'] and arg['type'] == 'array':
            template += f"\n\tcfl.writecfl(NAME + \'{name}\', {name})"

    template += "\n\n\tif DEBUG:"
    template += "\n\t\tprint(cmd_str)\n"
    template += "\n\n\tos.system(cmd_str)\n"

    # TODO: fix optional output (estdelay)
    if template_dict['has_output']:
        output_str = '\n\toutputs = '
        clean_str = ""
        return_str = '\n\treturn outputs'
        for arg in arg_list:
            name = arg['name']
            if not arg['input']:
                output_str += f"cfl.readcfl(NAME + '{name}'), "
                # clean_str += f"\n\tos.remove(\'{name}.hdr\')"
                # clean_str += f"\n\tos.remove(\'{name}.cfl\')"
        template += output_str.rstrip(', ')
        template += clean_str
        template += return_str

    return template.strip()


def write_tool_methods():
    """
    Autogenerate `tools.py` file which contains the BART tools
    """
    if not os.path.exists(TOOLS_PATH):
        with open(TOOLS_PATH, 'w+'):
            pass
    template_str = 'from ..utils import cfl\nimport os\nimport tempfile as tmp\n\n'
    template_str += "BART_PATH=os.environ['TOOLBOX_PATH'] + '/bart'\n"
    template_str += "DEBUG=False\n"
    template_str += "NAME=tmp.NamedTemporaryFile().name\n\n"
    template_str += "def set_debug(status):\n\tglobal DEBUG\n\tDEBUG=status\n\n\n"
    tool_lst = get_tools()[4:]
    for tool in tool_lst:
        template_str += create_template(tool)
        template_str += '\n\n'
    template_str = re.sub('\t', '    ', template_str)
    with open(TOOLS_PATH, 'w+') as f:
        f.write(template_str)

if __name__ == '__main__':
    write_tool_methods()

    