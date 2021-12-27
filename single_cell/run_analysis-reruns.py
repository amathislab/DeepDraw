# Kai Sandbrink
# 2021-10-31 
# Wrapper script to allow for batch reruns, in particular suitable analysis reruns

from controls_main import main

#method = 'parallel' # one of ['sequential', 'parallel']
method = 'sequential'

# %% TO RUN IN PARALLEL

if method == 'parallel':

    import multiprocessing as mp

    ''' #NOT INCLUDING RESULTS
    parsets = [
        (False, False, True, False, ['S'], ['task'], 301),
        (False, False, True, False, ['ST'], ['task'], 307),
        (False, False, True, False, ['LSTM'], ['task'], 315),
        (False, False, True, False, ['S'], ['regression'], 312),
        (False, False, True, False, ['ST'], ['regression'], 313),
        (False, False, True, False, ['LSTM'], ['regression'], 315),
    ]
    '''

    #INCLUDING RESULTS
    parsets = [
        (False, False, True, False, ['S'], ['task'], 301),
        (False, False, True, False, ['ST'], ['task'], 307),
        (False, False, True, False, ['LSTM'], ['task'], 315),
        #(False, True, True, False, ['S'], ['regression'], 312),
        #(False, True, True, False, ['ST'], ['regression'], 313),
        #(False, True, True, False, ['LSTM'], ['regression'], 315),
        (False, False, True, False, ['S'], ['regression'], 301),
        (False, False, True, False, ['ST'], ['regression'], 307),
        (False, False, True, False, ['LSTM'], ['regression'], 315),
    ]

    running_tasks = []
    for parset in parsets:
        print("Starting process for ", parset)
        running_tasks.append(mp.Process(target=main, args=parset))
    
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()

else:
    print("running main for S TASK models...")
    main(False, False, True, False, ['S'], tasks=['task'], expid=301)
    print("running main for ST TASK models...")
    main(False, False, True, False, ['ST'], tasks=['task'], expid=307)
    print("running main for LSTM TASK models...")
    main(False, False, True, False, ['LSTM'], tasks=['task'], expid=315)
    print("running main for S REG models...")
    main(False, False, True, False, ['S'], tasks=['regression'], expid=301)
    print("running main for ST REG models...")
    main(False, False, True, False, ['ST'], tasks=['regression'], expid=307)
    print("running main for LSTM REG models...")
    main(False, False, True, False, ['LSTM'], tasks=['regression'], expid=315)