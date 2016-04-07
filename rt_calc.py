#import sav_gol_filt as sg_filt
import tables, time

def example_run(): 
    # Load HDF file
    hdf = tables.openFile('grom20160201_04_te4048.hdf')

    #Extract go_cue_indices in units of hdf file row number
    go_cue_ix = np.array([hdf.root.task_msgs[j-3]['time'] for j, i in enumerate(hdf.root.task_msgs) if i['msg']=='reward'])
    
    # Calculate filtered velocity and 'velocity mag. in target direction'
    filt_vel, vel_bins, vel_in_targ_dir = get_cursor_velocity(hdf, go_cue_ix, .1, 2., use_filt_vel=False)

    # Calculate 'RT' from vel_in_targ_direction
    kin_feat = get_kin_sig_shenoy_method(vel_in_targ_dir.T, vel_bins, perc=.2, start_tm = .1)

    #PLot first 15 trials in a row
    for n in range(15):
        plt.plot(vel_in_targ_dir[:, n], '.-')
        plt.plot(kin_feat[n, 0], vel_in_targ_dir[int(kin_feat[n,0]), n], '.', markersize=10)
        plt.show()
        time.sleep(1.)

    return kin_feat

def get_cursor_velocity(hdf, go_cue_ix, before_cue_time, after_cue_time, fs=60., use_filt_vel=True):
    '''
    hdf file -- task file generated from bmi3d
    go_cue_ix -- list of go cue indices (units of hdf file row numbers)
    before_cue_time -- time before go cue to inclue in trial (units of sec)
    after_cue_time -- time after go cue to include in trial (units of sec)

    returns a time x (x,y) x trials filtered velocity array
    '''

    ix = np.arange(-1*before_cue_time*fs, after_cue_time*fs).astype(int)


    # Get trial trajectory: 
    cursor = []
    target = []
    for g in go_cue_ix:
        try:
            #Get cursor
            cursor.append(hdf.root.task[ix+g]['cursor'][:, [0, 2]])
            # plt.plot(hdf.root.task[ix+g]['cursor'][:, 0]), hdf.root.task[ix+g]['cursor'][:, 2])
            # plt.plot(hdf.root.task[ix+g]['cursor'][0, 0]), hdf.root.task[ix+g]['cursor'][0, 2], '.', markersize=20)

            #Get target:
            target.append(hdf.root.task[g+4]['target'][[0, 2]])

        except:
            print 'skipping index: ', g, ' -- too close to beginning or end of file'
    cursor = np.dstack((cursor))    # time x (x,y) x trial
    target = np.vstack((target)).T #  (x,y) x trial

    dt = 1./fs
    vel = np.diff(cursor,axis=0)/dt

    #Filter velocity: 
    #filt_vel = sg_filt.savgol_filter(vel, 9, 5, axis=0)
    vel_bins = np.linspace(-1*before_cue_time, after_cue_time, vel.shape[0])

    mag = np.linalg.norm(target, axis=0)
    mag_arr = np.vstack((np.array([mag]), np.array([mag])))
    unit_targ = target/mag_arr
    mag_mat = np.tile(unit_targ, (vel.shape[0], 1, 1))

    #Dot prod of velocity onto unitary v
    if use_filt_vel:
        vel_in_targ_dir = np.sum(np.multiply(mag_mat, filt_vel), axis=1)
    else:
        vel_in_targ_dir = np.sum(np.multiply(mag_mat, vel), axis=1)

    return filt_vel, vel_bins, vel_in_targ_dir

def get_kin_sig_shenoy_method(kin_sig, bins, perc=.2, start_tm = .1):
    '''
    input:
        kin_sig: trials x time array corresponding to velocity of the cursor
        perc: reaction time is calculated by determining the time point at which the 
            kin_sig crosses a specific threshold. This threshold is defined by finding all
            the local maxima, then picking the highest of those (not the same as taking the 
            global maxima, because I want points that have a derivative of zero). 
            
            Then I take the point of the highest local maxima and step backwards until I cross 
            perc*highest_local_max value. If I never cross that threshold then I print a 
            statement and just take the beginning index of the kin_sig as 'rt'. This happens once
            or twice in my entire kinarm dataset. 

        start_tm: time from beginning of 'bins' of which to ignore any motion (e.g. if hold 
            time is 200 ms, and your kin_sig starts at the beginning of the hold time, set 
            start_tm = 0.2 to prevent micromovements in the hold time from being captured)

    output: 
        kin_feat : a trl x 3 array:
            column1 = RT in units of "bins" indices
            column2 = RT in units of time (bins[column1])
            column3 = index of max of kin_sig

    '''
    ntrials= kin_sig.shape[0]
    kin_feat = np.zeros((ntrials, 3))
    start_ix = int(start_tm * 60)
    #Iterate through trials
    for trl in range(ntrials):   
        spd = kin_sig[trl,:]

    
        d_spd = np.diff(spd)

        #Est. number of bins RT should come after: 
        
        #Find first cross from + --> -
        
        local_max_ind = np.array([i for i, s in enumerate(d_spd[:-1]) if scipy.logical_and(s>0, d_spd[i+1]<0)]) #derivative crosses zero w/ negative slope
        local_max_ind = local_max_ind[local_max_ind > start_ix]
        #How to choose: 
        if len(local_max_ind)>0:
            local_ind = np.argmax(spd[local_max_ind+1]) #choose the biggest
            bin_max = local_max_ind[local_ind]+1 #write down the time

        else:
            print ' no local maxima found -- using maximum speed point as starting pt'
            bin_max = np.argmax(spd)
       

        percent0 = spd[bin_max]*perc #Bottom Threshold
        ind = range(0, int(bin_max)) #ix: 0 - argmax_index
        rev_ind = ind[-1:0:-1] #Reverse
        rev_spd = spd[rev_ind]
        try:
            bin_rt = np.nonzero(rev_spd<percent0)[0][0]
        except:
            print 'never falls below percent of max speed'
            bin_rt = len(rev_spd)-1
        
        kin_feat[trl, 0] = rev_ind[bin_rt] #Index of 'RT'
        kin_feat[trl, 1] = bins[kin_feat[trl, 0]] #Actual time of 'RT'
        kin_feat[trl, 2] = bin_max 
    return kin_feat
