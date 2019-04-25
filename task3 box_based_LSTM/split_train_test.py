# split training and testing set by store the name of .txt files into a txt

import os

ICDAR_path = "../ICDAR_Dataset/"
task3_path = "../ICDAR_Dataset/task3-test(347p)/"
train_test_ratio = 400


def MoveFile(dir, file_train, file_test, wildcard, recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir + 'task3-test(347p)/task3-test(347p)/')
    num = 0
    for name in files:
        fullname = os.path.join(dir, name)
        if (os.path.isdir(fullname) & recursion):
            MoveFile(fullname, file_train, file_test, wildcard, recursion)
        else:
            for ext in exts:
                if (name.endswith(ext)):
                    if (num + 1) % (train_test_ratio + 1) != 0:
                        file_train.write(name + "\n")
                        num = num + 1
                    else:
                        file_test.write(name + "\n")
                        num = num + 1
                    break


def run():
    outfile_train = task3_path + "train.txt"
    outfile_test = task3_path + "test.txt"
    wildcard = ".txt"
    file_train = open(outfile_train, "w")
    file_test = open(outfile_test, "w")
    MoveFile(ICDAR_path, file_train, file_test, wildcard, 1)
    file_train.close()
    file_test.close()


if __name__ == '__main__':
    run()
