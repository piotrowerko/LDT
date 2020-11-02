from signal.signal import Signal

if __name__ == '__main__':

    myobject = Signal('ztpi')
    # print object name
    print(myobject)
    #import data
    myobject.load_data('./ztpi_data.txt')
    #return LSF object
    print(f'{myobject.lsq_resutls()}')
    # return Soft_L1 fit object
    print(f'{myobject.softL1_results()}')
    # return Huber fit object
    print(f'{myobject.huber_results()}')

