import sav_gol_filt as sg_filt
import tables, time
import numpy as np
import scipy
import matplotlib.pyplot as plt

def compute_rt_per_trial_FreeChoiceTask(hdf_file): 
    # Load HDF file
    hdf = tables.openFile(hdf_file)

    #Extract go_cue_indices in units of hdf file row number
    go_cue_ix = np.array([hdf.root.task_msgs[j-3]['time'] for j, i in enumerate(hdf.root.task_msgs) if i['msg']=='check_reward'])
    
    # Calculate filtered velocity and 'velocity mag. in target direction'
    filt_vel, total_vel, vel_bins = get_cursor_velocity(hdf, go_cue_ix, 0., 2., use_filt_vel=False)

    ## Calculate 'RT' from vel_in_targ_direction: use with get_cusor_velocity_in_targ_dir
    #kin_feat = get_kin_sig_shenoy_method(vel_in_targ_dir.T, vel_bins, perc=.2, start_tm = .1)
    #kin_feat = get_rt(total_vel.T, vel_bins, vel_thres = 0.1)
    kin_feat = get_rt_change_deriv(total_vel.T, vel_bins, d_vel_thres = 0.3, fs=60)
    
    '''
    #PLot first 5 trials in a row
    for n in range(1):
        plt.plot(total_vel[:, n], '.-')
        plt.plot(kin_feat[n, :][0], total_vel[int(kin_feat[n,:][0]), n], '.', markersize=10)
        plt.show()
        time.sleep(1.)
    '''
    hdf.close()
    
    return kin_feat[:,1], total_vel

def compute_rt_per_trial_StressTask(hdf_file, deriv_vel_thres): 
    # Load HDF file
    hdf = tables.openFile(hdf_file)

    #Extract go_cue_indices in units of hdf file row number
    go_cue_ix = np.array([hdf.root.task_msgs[j-3]['time'] for j, i in enumerate(hdf.root.task_msgs) if i['msg']=='check_reward'])
    state = hdf.root.task_msgs[:]['msg']
    state_time = hdf.root.task_msgs[:]['time']
    stress_type = hdf.root.task[:]['stress_trial']
    ind_check_reward_states = np.ravel(np.nonzero(state == 'check_reward'))
    successful_stress_or_not = np.ravel(stress_type[state_time[ind_check_reward_states]])


    # Calculate filtered velocity and 'velocity mag. in target direction'
    filt_vel, total_vel, vel_bins = get_cursor_velocity(hdf, go_cue_ix, 0., 2., use_filt_vel=False)

    ## Calculate 'RT' from vel_in_targ_direction: use with get_cusor_velocity_in_targ_dir
    #kin_feat = get_kin_sig_shenoy_method(vel_in_targ_dir.T, vel_bins, perc=.2, start_tm = .1)
    #kin_feat = get_rt(total_vel.T, vel_bins, vel_thres = 0.1)
    kin_feat = get_rt_change_deriv(total_vel.T, vel_bins, d_vel_thres = deriv_vel_thres, fs=60)
    
    '''
    #PLot first 5 trials in a row
    for n in range(1):
        plt.plot(total_vel[:, n], '.-')
        plt.plot(kin_feat[n, :][0], total_vel[int(kin_feat[n,:][0]), n], '.', markersize=10)
        plt.show()
        time.sleep(1.)
    '''
    hdf.close()
    
    return kin_feat[:,1], total_vel, successful_stress_or_not

def compute_rt_per_trial_CenterOut(hdf_file, method, thres, plot_results): 
    # Load HDF file
    hdf = tables.openFile(hdf_file)

    #Extract go_cue_indices in units of hdf file row number
    go_cue_ix = np.array([hdf.root.task_msgs[j-3]['time'] for j, i in enumerate(hdf.root.task_msgs) if i['msg']=='reward'])
    
    # Calculate filtered velocity and 'velocity mag. in target direction'
    filt_vel, total_vel, vel_bins, skipped_indices = get_cursor_velocity(hdf, go_cue_ix, 0., 2., use_filt_vel=False)

    ## Calculate 'RT' from vel_in_targ_direction: use with get_cusor_velocity_in_targ_dir
    #kin_feat = get_kin_sig_shenoy_method(vel_in_targ_dir.T, vel_bins, perc=.2, start_tm = .1)
    if method==1:
        kin_feat = get_rt(total_vel.T, vel_bins, vel_thres = thres)
    elif method==2:
        kin_feat = get_rt_change_deriv(total_vel.T, vel_bins, d_vel_thres = thres, fs=60)
    else:
        print "Did not choose valid method"
        kin_feat = np.zeros((len(total_vel),2))
    
    
    #PLot first 5 trials in a row
    if plot_results:
        for n in range(5):
            plt.plot(total_vel[:, n], '.-')
            plt.plot(kin_feat[n, :][0], total_vel[int(kin_feat[n,:][0]), n], '.', markersize=10)
            plt.show()
            time.sleep(1.)
    
    return kin_feat[:,1], total_vel, skipped_indices

def get_cursor_velocity(hdf, go_cue_ix, before_cue_time, after_cue_time, fs=60., use_filt_vel=True):
    '''
    hdf file -- task file generated from bmi3d
    go_cue_ix -- list of go cue indices (units of hdf file row numbers)
    before_cue_time -- time before go cue to inclue in trial (units of sec)
    after_cue_time -- time after go cue to include in trial (units of sec)

    returns a time x (x,y) x trials filtered velocity array
    '''

    ix = np.arange(-1*before_cue_time*fs, after_cue_time*fs).astype(int)
    skipped_indices = np.array([])

    # Get trial trajectory: 
    cursor = []
    for k,g in enumerate(go_cue_ix):
        try:
            #Get cursor
            cursor.append(hdf.root.task[ix+g]['cursor'][:, [0, 2]])

        except:
            print 'skipping index: ', g, ' -- too close to beginning or end of file'
            skipped_indices = np.append(skipped_indices, k)

    cursor = np.dstack((cursor))    # time x (x,y) x trial
    
    dt = 1./fs
    vel = np.diff(cursor,axis=0)/dt

    #Filter velocity: 
    if use_filt_vel:
        filt_vel = sg_filt.savgol_filter(vel, 9, 5, axis=0)
    else:
        filt_vel = vel
    total_vel = np.zeros((int(filt_vel.shape[0]),int(filt_vel.shape[2])))
    for n in range(int(filt_vel.shape[2])):
        total_vel[:,n] = np.sqrt(filt_vel[:,0,n]**2 + filt_vel[:,1,n]**2)

    vel_bins = np.linspace(-1*before_cue_time, after_cue_time, vel.shape[0])

    return filt_vel, total_vel, vel_bins, skipped_indices

def get_cursor_velocity_in_targ_dir(hdf, go_cue_ix, before_cue_time, after_cue_time, fs=60., use_filt_vel=True):
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

def get_rt(kin_sig, bins, vel_thres = 0.):
    '''
    input:
        kin_sig: trials x time array corresponding to velocity of the cursor
        
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
    kin_feat = np.zeros((ntrials, 2))
    
    #Iterate through trials
    for trl in range(ntrials):   
        spd = kin_sig[trl,:]
        bin_rt = np.ravel(np.nonzero(np.greater(spd,vel_thres)))[0]
        
        kin_feat[trl, 0] = bin_rt #Index of 'RT'
        kin_feat[trl, 1] = bins[kin_feat[trl, 0]] #Actual time of 'RT'
    return kin_feat

def get_rt_change_deriv(kin_sig, bins, d_vel_thres = 0., fs = 60):
    '''
    input:
        kin_sig: trials x time array corresponding to velocity of the cursor
        
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
    kin_feat = np.zeros((ntrials, 2))
    
    #Iterate through trials
    for trl in range(ntrials):   
        spd = kin_sig[trl,:]

        dt = 1./fs
        d_spd = np.diff(spd,axis=0)/dt
        
        if len(np.ravel(np.nonzero(np.greater(d_spd,d_vel_thres))))==0:
            bin_rt = 0
        else:
            bin_rt = np.ravel(np.nonzero(np.greater(d_spd,d_vel_thres)))[0]
        
        kin_feat[trl, 0] = bin_rt + 1 #Index of 'RT'
        kin_feat[trl, 1] = bins[kin_feat[trl, 0]] #Actual time of 'RT'
    return kin_feat


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
