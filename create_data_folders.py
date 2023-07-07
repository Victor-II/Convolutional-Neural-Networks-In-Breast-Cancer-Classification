import splitfolders

input_folder = 'Path/To/BreakHis'
output_folder = 'Path/To/BreakHis_split'

if __name__ == '__main__':

    splitfolders.ratio(input_folder,
                    output=output_folder,
                    seed=42,
                    ratio=(0.8, 0.1, 0.1))

    
    
