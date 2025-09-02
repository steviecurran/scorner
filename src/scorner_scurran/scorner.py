import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import statistics
from matplotlib import pyplot as plt, patches

def scorner(
        data,
        cols= None,
        true_values=[], 
        true_style = ['dotted','2','k'],
        test_run = False, 
        test_no = 500,
        bins=20,
        cmap = 'rainbow',
        fs=12,
        xlabel_off = False,
        ylabel_off = False,
        height = 1.5,
        aspect = 1,
        aw = 2, 
        ap = 7, 
        point_s = 2,
        point_c = 'grey',
        point_top = False,
        lrot = 0, 
        min_space = 5, 
        dens = True, 
        dens_alpha = 1,
        con = False, 
        con_levels = 4,
        con_thresh = 0.2,
        show_stats = False,
        stats_in = True,
        leg_scale = 0.8,
        leg_dps = 2,
        leg_alpha = 0.5,
        leg_fc = 'w',
        tlp = 0.3, 
        hc = False, 
        plot_form = 'png', 
        plot_name = 'scorner'
):

    """
    Produces a similar plot to  corner.py, which shows the
    projections of data in a multi-dimensional space. This uses
    seaborn to produce a more adjustable, 'publication ready' plot,
    e.g. through variable axis thickness, the use of colour maps,
    line styles, choice of contours/density maps and order of plotting
    in relation to scatter plot. Also included is an option to save a
    hard copy in the required format.

    Requires
             import numpy as np
             import pandas as pd
             import matplotlib.pyplot as plt
             import seaborn as sns
             import matplotlib.ticker as ticker
    
    Basic usage
             import scorner as sc
             sc.scorner(data)
    where data is an array or dataframe
    
    Parameters
    ----------

    data : obj

        The data in the form of an array or dataframe. 

    cols: list

        If data input as array this specifies the column names of the
        dataframe (required by seaborn).

    true_values: list

        The mean values used to generate the data. Default is not to
        show.

    true_style : list
        Line style, width and colour if showing true_values
    
    test_run : bool

        If True a test using test_no rows of the dataframe will be
        run.  Since sns.kdeplot can take a while to run, this is
        advisable until the plot appearance is satisfactory. Default
        is to run the whole dataset.

    test_no : int

        The number of rows which test_run will use. Default is 500
     
    bins : int

        The number of histogram bins

    cmap : str

        The colour map of the density plot. Default is 'rainbow'
        https://matplotlib.org/stable/users/explain/colors/colormaps.html

    fs : float

        Specifies the font size, which scales with the height of the plot

    xlabel_off : bool

        If True this removes the x-axis labels. If False the dataframe column
        names are used

    ylabel_off : bool

        If True this removes the y-axis labels. If False the dataframe column
        names are used

    height : float

        The height of the plot according to seaborn. The default is 1.5

    aspect : float

        The aspect of the plot according to seaborn

    aw : float

        Axis widths, the default is 2

    aw : float

        Axis padding, the default is 7

    point_s : float

        Point size in scatter plots

    point_c:

        Point colour in scatter plots

    lrot : float

        Axis label rotation angle, default is 0

    min_space : int

        Number of minor ticks per major tick interval. Default is 5

    dens : bol

       If True this will plot a density maps using the specified colour map.
       Default is True

    dens_alpha : float

       The transparency of the density map. Default is fully opaque (alpha = 1)

    con : bol

      If True this will show a contour map. Default is no contour map

    con_levels : int

      The number of contours

    con_thresh : float

      The lowest contour value

    show_stats : bool

      Show the mean and standard deviation in the histogram. Default is False

    stats_in : bool

      If True (default) the above legend is shown within the histograms. If
      False it is show above the histograms

    leg_scale: float

      If show_stats=True, the size of the legend font in relation to the tick font

    leg_dps : int

      If show_stats=True, the number of decimal places to show these to

    leg_alpha : float

      Transparency of legend box if stats_in = True. Default is 0.5

    leg_fc : str

      Colour of legend box if stats_in = True, Default is 'w'

    tlp : float

      The plt.tight_layout pad value, the default is 0.3

    hc : bool

      hc = True saves a hard copy. The default is False

    plot_form : str

      The format of the hard copy, the default is 'png'

    plot_name : str

      The name of the hard copy, the default is 'scorner'

    """

    if isinstance(data, np.ndarray):
        if cols == None:
            cols = ["Col%d" %(i) for i in range(0,data.shape[1])]
        df = pd.DataFrame(data, columns=cols)
            
    else:
        df = data.copy()
        cols = df.columns
  
    ndim = len(cols)
    boxes = [x for x in range(0,ndim**2)]
    boxes = np.reshape(boxes,(ndim,-1)); 
    diags = np.diag(boxes)
       
    #### TEST RUN #####
    if test_run == True:
        if test_no < len(df):
            top = df.head(int(test_no/2))#ndim*50)
            bot = df.tail(int(test_no/2))#ndim*50)
            df = pd.concat([top,bot], ignore_index=True)
            print("Testing on %d rows" %(test_no))
        else:
            print("Less than %d lines, no need to test" %(test_no))

    #### STATS #####
    s = []
    for i in range(0,ndim):
        mean = np.mean(df.iloc[:, [i]])
        sd = np.std(df.iloc[:, [i]], axis=0, ddof = 1) 
        sd = float(sd.values)# GETTING ALL THIS dtype: float64, B    1.019083 CRAP # sequoia
                
        s.append(diags[i]); s.append(mean); s.append(sd) 
    
    s = np.reshape(s,(-1,3))
    sdf = pd.DataFrame(s, columns=['box','mean','sd'])
    sdf['box'] = sdf['box'].astype(int)
    sdf['para'] = cols
    
    ############ PLOT SET-UP ##################
    font = fs*height/1.5
    plt.rcParams["axes.labelsize"] = font
    plt.rcParams["xtick.labelsize"] = font
    plt.rcParams["ytick.labelsize"] = font
    plt.rcParams['legend.handlelength'] = 0
      
    #### ORDER OF SCATTER PLOT AND DENSITY MAP ###
    if point_top == True:
        point_order = 2
    else:
        point_order = 0

    ### PLOTS ####
    g = sns.PairGrid(df, height=height, aspect=aspect, diag_sharey=False, corner=True)
    g.map_lower(sns.scatterplot,s=point_s,c=point_c,zorder=point_order)

    if dens == True:
        g.map_lower(sns.kdeplot,fill=True,cmap=cmap, alpha = dens_alpha,zorder=1)
        
    if con == True:
        g.map_lower(sns.kdeplot,levels=con_levels,thresh=con_thresh,zorder=1) 

    g.map_diag(sns.histplot,bins=bins,element="step",fill=False,zorder=1)
    
        
    panels = []
    for i, ax in enumerate(g.axes.flat):
        if str(ax) != 'None':
            plt.setp(ax.spines.values(), linewidth=aw)
            ax.spines['top'].set_visible(True);
            ax.spines['right'].set_visible(True); ax.spines['left'].set_visible(True)

            ax.tick_params(direction='in', pad = ap, length=4*height, width=0.8*aw, which='major',
                           right=True,top=True, labelrotation=lrot)
            ax.tick_params(direction='in', pad = ap, length=2*height, width=0.8*aw, which='minor',
                           right=True,top=True, labelrotation=lrot)

            x_maj = ax.get_xticks(); x_min = (x_maj[1] -  x_maj[0])/min_space
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_min))

            y_maj = ax.get_yticks(); y_min = (y_maj[1] -  y_maj[0])/min_space
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_min))

            if xlabel_off == True:
                ax.set(xlabel=None)

            if ylabel_off == True:
                ax.set(ylabel=None) 
                    
            x1, x2 = ax.get_xlim(); y1, y2 = ax.get_ylim()
            xpos = xpos = x1+(x2-x1)/16; ypos = y2-(y2-y1)/6;
            xskip = (x2-x1)/4; yskip = (y2-y1)/8

            panels.append(i)
        non_diags = [item for item in panels if item not in diags]
        
        vert = []
        for j in range(0,ndim):  
            tmp = [item for item in non_diags if item%ndim==j]
            vert = vert+tmp

        for j,d in enumerate(diags):
            if d == i:
                '''
                y1, y2 = ax.get_ylim()
                print(i,y2)
                #ax.set_ylim(y1,1.5*y2)
                plt.ylim(0,1.5*y2)
                print(i,y2)
                '''               
                para = sdf['para'].iloc[j]; para = para.strip('$')
                mean =  sdf['mean'].iloc[j]; sd =  sdf['sd'].iloc[j]
                
                if show_stats == True:   
                                      
                    dps = "{:.%df}" %(leg_dps) 
                    mean_d = dps.format(float(mean)); sd_d = dps.format(float(sd))
                    mtext = r"$\overline{%s} = %s\pm%s$" %(para,mean_d,sd_d)

                    if stats_in == True:
                        box_props = dict(boxstyle='square', facecolor = leg_fc, lw=0, alpha=leg_alpha)
                        ax.text(xpos,ypos,mtext,fontsize = leg_scale*font,c="k",zorder=2,
                                bbox=box_props, # STILL CAN'T GET OPACITY
                                fontweight='bold') # DOES NOTHING. STILL NOT HAPPY WITH THIS
                        '''
                        ax.plot([], [], ' ', label=r"%s" %(mtext),zorder = 2)
                        ax.legend(loc ='upper left',fontsize = leg_scale*font,
                              framealpha=1) # DOESN'T WORK EITHER
                        '''
                        #t.set_bbox(dict(fc='w',alpha=0.5,linewidth=0)) # NOT WORKING EITHER

                    else:
                        ax.set_title(mtext,fontsize = leg_scale*font,c="k") # COMPROMISE
                    
                      
            true_values = np.array(true_values)
                    
            if len(true_values) > 0:
                if len(true_values.shape) ==1:
                    for j,n in enumerate(non_diags):   
                        if n == i:
                            if non_diags[j]%ndim ==0:
                                horiz = int(non_diags[j]/ndim)
                                #print(horiz)
                            ax.axhline(true_values[horiz],linestyle=true_style[0],
                                       lw=true_style[1],color=true_style[2])

                    for j,n in enumerate(vert):
                        if n == i:
                            v = vert[j]%ndim
                            ax.axvline(true_values[v],linestyle=true_style[0],
                                       lw=true_style[1],color=true_style[2])
                
                else:
                    for t in range(0,true_values.shape[0]):
                        tv = true_values[t]
                        ts = true_style[t]; 

                        for j,n in enumerate(non_diags):   
                            if n == i:
                                if non_diags[j]%ndim ==0:
                                    horiz = int(non_diags[j]/ndim)
                                ax.axhline(tv[horiz],linestyle=ts[0],lw=ts[1],color=ts[2])

                        for j,n in enumerate(vert):
                            if n == i:
                                v = vert[j]%ndim
                                ax.axvline(tv[v],linestyle=ts[0],lw=ts[1],color=ts[2])
                    
    plt.tight_layout(pad = tlp)
    if hc == True:
        if test_run == True:
            plot = "%s_test.%s" %(plot_name,plot_form)
        else:
            plot = "%s.%s" %(plot_name,plot_form)

        plt.savefig(plot, format = "%s" %(plot_form));
        print("Plot written to", plot)
    plt.show()

