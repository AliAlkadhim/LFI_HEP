#fix nu, and plot
def plot_data(nu, NMLIST, 
              func=None,
              thetabins=THETABINS,
              thetamin=THETAMIN, 
              thetamax=THETAMAX,
              gfile='fig_data2_%s.png' % WHICH, 
              fgsize=(10, 6)):
    
    # make room for 6 sub-plots
    fig, ax = plt.subplots(nrows=2, 
                           ncols=3, 
                           figsize=fgsize)
    
    # padding
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(wspace=0.20)
    
    # use flatten() to convert a numpy array of 
    # shape (nrows, ncols) to a 1-d array. 
    ax = ax.flatten()
    
    for j, (N, M) in enumerate(NM_LIST):
        
        theta_counts, bb = hist_theta_for_NM_values(nu, N, M)
    
        ax[j].set_xlim(thetamin, thetamax)
        ax[j].set_ylim(0, 1)
        ax[j].set_xlabel(r'$\theta$', fontsize=FONTSIZE)
        ax[j].set_ylabel(r'$E(Z|\theta, \nu)$', fontsize=FONTSIZE)
        
        thetabins = (bb[1:]+bb[:-1])/2
        ax[j].plot(thetabins, theta_counts, 'b', lw=2, label='approx')
        
        if func:
            p, _ = func(nu, N, M)
            ax[j].plot(x, p, 'r', lw=2, label='model')
        
        ax[j].grid(True, which="both", linestyle='-')
        ax[j].text(10.1, 0.42, r'$N, M = %d, %d$' % (N, M), 
                   fontsize=FONTSIZE) 

        ax[j].text(10.1, 0.30, r'$\nu = %5.1f$' % nu, 
                   fontsize=FONTSIZE) 

        ax[j].legend(loc='upper right')
        
    # hide unused sub-plots
    for k in range(j+1, len(ax)):
        ax[k].set_visible(False)
    
    plt.tight_layout()
    print('saving..', gfile)
