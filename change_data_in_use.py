def change_data_in_use():
    onlyfiles = [f.split('_') for f in listdir('DATA/XU100-29022020/')]
    new_file_names = list()
    for i, filename in enumerate(onlyfiles):
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
    

    with open("DATA/in_use.txt", "w") as file:
        file.write('DATA/XU100-29022020/' + sett[ index_for_new_data - 1])

