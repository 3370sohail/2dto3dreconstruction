import os


def rename_move(src, dst, type):
    """
    function to make rgb d to a single folder and in the correct naming
    Args:
        src:
        dst:
        type:
    Returns:
    """
    path = os.path.abspath(src)
    file_names = os.listdir(path)
    count = 1
    for i in range(1, len(file_names) + 1, 10):
        file_name = file_names[i - 1]
        new_filename = 'File' + str(count)
        old_filename = "{}\{}".format(src, file_name)
        new_filename = "{}\{}.{}".format(dst, new_filename, type)
        print('old: ', old_filename)
        print('new:', new_filename)
        os.rename(old_filename, new_filename)
        count += 1
    print('Job Done!')


loc = r"C:\Users\sohai\Documents\Uni 2020\csc420\Project\06020\rgb"
locd = r"C:\Users\sohai\Documents\Uni 2020\csc420\Project\06020\depth"
loc2 = r"C:\Users\sohai\Documents\Uni 2020\csc420\Project\other3"

rename_move(loc, loc2, 'jpg')
rename_move(locd, loc2, 'png')
