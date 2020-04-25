from lung_segmentor import CtLungSegmentor
import sys 
import os 


if __name__ == '__main__':
    
    ct_segmentor = CtLungSegmentor()


    path = sys.argv[1]

    if os.path.isdir(path):
        print('Processing directory ' + path +'\n')
        ct_segmentor.process_dir(path)
    
    elif os.path.isfile(path):

        if path.lower().endswith('.nii.gz'):
            print('Processing file ' + path + '\n')

            ct_segmentor.process_file(path)

        else:
            print('\nOnly .nii or .nii.gz files are supported\n')

    else:
        print('File or directory not found: {}'.format(path))
        exit(2)