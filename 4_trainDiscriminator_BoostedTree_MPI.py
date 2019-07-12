from mpi4py import MPI
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import GradientBoostingClassifier
import GMCluster as gml
import discr_utils as du
from matplotlib import cm
from operator import add, sub
import pickle
import sys
from matplotlib import rc

def cm2inch(value):
	return value/2.54

#Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
rootRank = 0

###########################################
# User-defined parameters
###########################################
#Selected IMs for the prediction
IM_name = ['PGA', 'PGV', 'AI', 'pSA_0.1', 'pSA_1.0', 'pSA_3.0']

#Number of cross-validation folders
K = 5

#Number of trees
N_estimator = [2000]

#Model name
model_name = 'discriminator.mdl'

#Tested features max
learning_rate = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
###########################################


#######################################################
# Preparation work
#######################################################
if rank == rootRank:
    #Number of workers
    n_worker = size-1
    
    print('Load data...')
    sys.stdout.flush()

#Import data    
X = np.load('./data/X_train.npy')
y = np.load('./data/y_train.npy')
lbl_GM = gml.loadLabelDict('label_dict.csv', reverse=True)
lbl_GM = list(lbl_GM.keys())
IM_dict = gml.loadIMDict_trainData('IM_dict_train.csv')
IM_ID = [int(IM_dict[x]) for x in IM_name]

#Split the dataset (making sure all labels are present in training set)
if rank == rootRank:
    print('Create and broadcast ' + str(K) + ' cross-validation folders...')
    sys.stdout.flush()
    ind_K = du.createKFold(K, y, lbl_GM)
else:
    ind_K = None
    
#Broadcast ind_K
comm.barrier()
ind_K = comm.bcast(ind_K, root=rootRank)

#Folder and file name
dir_path = './BT_Discriminator/'
if rank == rootRank:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


#######################################################
# Main thread of the manager
#######################################################
if rank == rootRank:
    #Create job list
    job = []
    for i_lr in range(len(learning_rate)):
        for i_IM in  range(len(IM_name)):
            for i_est in range(len(N_estimator)):
                for i in range(K):
                    job.append([i_lr, i_IM, i_est, i])
         
    #Send initial jobs
    n_job = len(job)
    id_next_job = 0
    worker_lib = np.zeros((n_worker))
    for i in range(1, size):
        #Send index of job
        comm.send(job[i-1], dest=i, tag=0)
        id_next_job = id_next_job + 1
    
    #Initialize receiver matrices and variables
    job_compl = 0
    N_best = 0
    IM_best = ''
    lr_best = -1.0
    CV_err = [[[[[] for l in range(K)] for k in range(len(N_estimator))] for j in range(len(IM_name))] for i in range(len(learning_rate))]
    
    print('Start the ' + str(K) + '-Fold crossvalidation...')
    sys.stdout.flush()
    
    CV_err_mean = [[[[] for k in range(len(N_estimator))] for j in range(len(IM_name))] for i in range(len(learning_rate))]
    CV_err_std = [[[[] for k in range(len(N_estimator))] for j in range(len(IM_name))] for i in range(len(learning_rate))]
    
    #Do the rest
    while(True):
        #Receive the rank of the worker having completed job
        ind_free = -1
        ind_free = comm.recv(source=MPI.ANY_SOURCE, tag=1)
        job_compl = job_compl + 1
        
        #Receive job characteristics
        ind_job = []
        ind_job = comm.recv(source=ind_free, tag=2)
        
        #Receive CV_error_mean of the completed job
        CV_i = -1.0
        CV_i = comm.recv(source=ind_free, tag=3)
        
        #Store results
        CV_err[ind_job[0]][ind_job[1]][ind_job[2]][ind_job[3]] = CV_i
        
        print('\t Core \t' + str(ind_free) + ':\t' + str(job_compl) + '/' + str(n_job) )
        sys.stdout.flush()
        
        #Check if all jobs are completed
        if id_next_job >= n_job:
            comm.send([-1, -1, -1, -1], dest=ind_free, tag=0)
            worker_lib[ind_free-1] = 1
            
        #If not send next one
        else:
            comm.send(job[id_next_job], dest=ind_free, tag=0)
            id_next_job = id_next_job + 1
            
        #If all workers are free, save results and close program
        if np.sum(worker_lib) == n_worker:
            print('Compute the CV metrics and plot results...')
            sys.stdout.flush()
            break 
        
    #Once all the job are done, compute the CV metrics (mean and std)
    lr_best = -1.
    IM_best = ''
    N_best = -1
    CV_min = 1000000.0
    for i_lr in range(len(learning_rate)):
        for i_IM in range(len(IM_name)):
            for i_est in range(len(N_estimator)):
                CV_err_mean[i_lr][i_IM][i_est] = np.mean(CV_err[i_lr][i_IM][i_est])
                CV_err_std[i_lr][i_IM][i_est] = np.std(CV_err[i_lr][i_IM][i_est])
                if CV_min > CV_err_mean[i_lr][i_IM][i_est]:
                    CV_min = CV_err_mean[i_lr][i_IM][i_est]
                    lr_best = learning_rate[i_lr]
                    IM_best = IM_name[i_IM]
                    N_best = N_estimator[i_est]
    
    #Train model with selected hyperparameters
    X_test = np.load('./data/X_test.npy')
    y_test = np.load('./data/y_test.npy')
    X = X[:, :, int(IM_dict[IM_best])]
    X_test = X_test[:, :, int(IM_dict[IM_best])]
    
    #Create and fit the model
    bestModel = GradientBoostingClassifier(learning_rate=lr_best, n_estimators=N_best, min_samples_split=4, subsample=0.5, min_samples_leaf=2, max_depth=4, max_features=42, verbose=0, validation_fraction=0.1, n_iter_no_change=5)
    bestModel = bestModel.fit(X, y)
        
    #Test on validation set
    y_test_hat = bestModel.predict(X_test)
        
    #Estimate the error
    accuracy_count = 0
    for k in range(len(y_test)):
        if y_test[k] == y_test_hat[k]:
            accuracy_count = accuracy_count + 1
    accuracy = 100.*float(accuracy_count)/float(len(y_test))
    error_test = 1.0-accuracy/100.
    
    #Save results
    np.save(dir_path + 'lr.npy', np.array(learning_rate))
    np.save(dir_path + 'im_name.npy', np.array(IM_name))
    np.save(dir_path + 'N_est.npy', N_estimator)
    np.save(dir_path + 'CV_mean.npy', np.array(CV_err_mean))
    np.save(dir_path + 'CV_std.npy', np.array(CV_err_std))
    np.save(dir_path + 'lr_best.npy', np.array(lr_best))
    np.save(dir_path + 'N_best.npy', np.array(N_best))
    np.save(dir_path + 'im_best.npy', np.array(IM_best))
    np.save(dir_path + 'error_test.npy', np.array(error_test))
    
    
    #Plot the results
    IM_ID_norm = [x/len(IM_ID) for x in IM_ID]
    cm_IM = cm.get_cmap('tab10')
    IM_col = cm_IM([IM_ID_norm])
    '''
    CV_err_mean = np.asarray(CV_err_mean)
    CV_err_std = np.asarray(CV_err_std)

    rc('text', usetex=True)
    rc('font', family='serif', size=13)

    fig, ax = plt.subplots()
    fig.set_size_inches(cm2inch(16), cm2inch(11))
    for i_IM in range(len(IM_name)):
        lbl_str = IM_name[i_IM]# + ', ' + str(learning_rate[i_lr])
        ax.plot(learning_rate, CV_err_mean[:, i_IM, :], label=lbl_str, color=IM_col[0, i_IM, :])
        ax.fill_between(learning_rate, np.squeeze(CV_err_mean[:, i_IM] + CV_err_std[:, i_IM]), np.squeeze(CV_err_mean[:, i_IM] - CV_err_std[:, i_IM]),  color=IM_col[0, i_IM, :], alpha=0.3)
    ax.scatter(lr_best, error_test, s=60, marker='d', color='black', label='Selected model (' + IM_best + ')')
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xscale('log')
    ax.set_xlim([max(learning_rate), min(learning_rate)])
    lgd = ax.legend(bbox_to_anchor=(1.02, 1), loc=2, ncol=1, borderaxespad=0., fontsize=11)
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Cross-validation error (inaccuracy)')
    plt.savefig(dir_path + 'CV_BT_' + str(i_lr) + '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')
    '''
    #Save the model
    model_path = dir_path + model_name
    pickle.dump(bestModel, open(model_path, 'wb'))


#######################################################
# Main thread of the workers
#######################################################
else: 
    #Receive initial index of image
    job_i = [-1, -1, -1, -1]
    job_i = comm.recv(source=rootRank, tag=0)
    
    while not job_i[0] == -1:        
        #Perform job
        #Create training folders
        X_train, y_train, X_val, y_val = du.createTrainValDataset(X, y, ind_K, job_i[3], IM_ID[job_i[1]])

        #Create and fit the model
        BT = GradientBoostingClassifier(learning_rate=learning_rate[job_i[0]], n_estimators=N_estimator[job_i[2]], min_samples_split=4, min_samples_leaf=2, subsample=0.5, max_depth=4, max_features=42, verbose=0, validation_fraction=0.1, n_iter_no_change=5)
        BT = BT.fit(X_train, y_train)
        
        #Test on validation set
        y_val_hat = BT.predict(X_val)
        
        #Estimate the error
        accuracy_count = 0
        for k in range(len(y_val)):
            if y_val[k] == y_val_hat[k]:
                accuracy_count = accuracy_count + 1
        accuracy = 100.*float(accuracy_count)/float(len(y_val))
        CV_i = 1.0-accuracy/100.
        
        #Send results
        comm.send(rank, dest=rootRank, tag=1)
        comm.send(job_i, dest=rootRank, tag=2)
        comm.send(CV_i, dest=rootRank, tag=3)
        
        #Receive next job ID
        job_i = comm.recv(source=rootRank, tag=0)


    
