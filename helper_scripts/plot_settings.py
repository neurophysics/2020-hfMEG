import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt

####################################
# apply some settings for plotting #
####################################
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['axes.unicode_minus'] = False

cmap = 'plasma'
color1 = '#e66101'.upper()
color2 = '#5e3c99'.upper()

pgf_with_latex = {
    "pgf.texsystem": "lualatex",     # Use xetex for processing
    "text.usetex": True,            # use LaTeX to write all text
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([
        r'\usepackage[sc]{mathpazo}',
        r'\usepackage{sfmath}',
        r'\usepackage{xcolor}',     # xcolor for colours
        r'\usepackage[super]{nth}', # nth for counts
        r'\usepackage{textgreek}',
        r'\usepackage{amsmath}',
        r'\usepackage{marvosym}',
        r'\usepackage{graphicx}',
        r'\usepackage{fontspec}'
        r'\setmainfont{Source Sans Pro}'
    ])
}

mpl.rcParams.update(pgf_with_latex)

blind_ax = dict(top=False, bottom=False, left=False, right=False,
        labelleft=False, labelright=False, labeltop=False, labelbottom=False)
