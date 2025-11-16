import matplotlib.pyplot as plt

def SetRC(
        style="default",font:str="serif",fontSize:int=26,
        axTitleSize:int=None,axLabelSize:int=None,
        xTickLabelSize:int=None,yTickLabelSize:int=None,
        legendFontSize:int=None,titleFontSize:int=None,
        legendLabelSpacing:int=None
    ):
    """
    Set up the matplotlib runtime configuration (RC).
    The default RC uses the "default" style, the "serif" font-style, and the same font-size of 26 for all text elements. 

    Refs:
    - https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    - https://stackoverflow.com/questions/52673444/matplotlib-rcparams-not-recognizing-times-new-roman-mac-high-sierra

    Args:
        style (str, optional): Defaults to "default".
        font (str, optional): Defaults to "serif".
        fontSize (int, optional): Defaults to 24.
        axTitleSize (int, optional): Defaults "fontSize".
        axLabelSize (int, optional): Defaults "fontSize".
        xTickLabelSize (int, optional): Defaults "fontSize".
        yTickLabelSize (int, optional): Defaults "fontSize".
        legendFontSize (int, optional): Defaults "fontSize".
        titleFontSize (int, optional): Defaults "fontSize".
        legendLabelSpacing (int, optional): Defaults to None.
    """

    if axTitleSize is None: axTitleSize = fontSize
    if axLabelSize is None: axLabelSize = fontSize
    if xTickLabelSize is None: xTickLabelSize = fontSize
    if yTickLabelSize is None: yTickLabelSize = fontSize
    if legendFontSize is None: legendFontSize = fontSize
    if titleFontSize is None: titleFontSize = fontSize

    plt.style.use(style)
    plt.rc('font', size=fontSize,family=font) # controls default text sizes
    plt.rc('axes', titlesize=axTitleSize)     # fontsize of the axes title    
    plt.rc('axes', labelsize=axLabelSize)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=xTickLabelSize) # fontsize of the tick labels
    plt.rc('ytick', labelsize=yTickLabelSize) # fontsize of the tick labels
    plt.rc('legend', fontsize=legendFontSize) # legend fontsize
    plt.rc('figure', titlesize=titleFontSize) # fontsize of the figure title

    if legendLabelSpacing is not None: 
        plt.rc('legend',labelspacing=legendLabelSpacing) # legend fontsize

    return 