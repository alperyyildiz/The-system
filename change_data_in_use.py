def change_data_in_use():
    dtype = input('Multiple or Single')
    data_storage_dir = 'DATA/XU100-29022020/' + dtype + '/'
    data_in_use_dir = 'DATA/in_use_info.txt'
    onlyfiles = [f.split('_') for f in listdir( data_storage_dir )]

    new_file_names = list()
    print(onlyfiles)
    for i, filename in enumerate(onlyfiles):
        name  = join(filename[:-1], '_')
        print(name)
        new_file_names.append(join(filename[:-1], '_'))
    sett = set(new_file_names)

    indexes = [i + 1 for i in range(len(sett))]

    result = zip(indexes, sett)
    sett = list(sett)
    
    print('Choose data to be in use, reply with index num')

    for item in sett:
        print(sett.index(item)+1,item, sep=' -- ')
        print('\n')


    
    index_for_new_data = int(input('\n'))
    elem = sett[ index_for_new_data - 1]

    print('ELEM ELEM ELEM')
    print(elem)
    print('ELEM ELEM ELEM')
    
    with open(data_storage_dir + elem + '_.txt', "r") as f:
        info = f.read()
    with open(data_in_use_dir , 'w') as f:
        f.write(info)







