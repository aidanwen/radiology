from shutil import copyfile
import csv
yes = []
no = []

with open('stage_1_train_labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            imgID = row[0]
            x = row[1]
            y = row[2]
            width = row[3]
            height = row[4]
            target = row[5]
            if target:
                yes.append(imgID)
            else:
                no.append(imgID)
            line_count += 1
print(yes)

for i in range(len(yes)):
    #move to
    copyfile('stage_1_train_images/' + yes[i] + '.dcm', 'Xception-with-Your-Own-Dataset/pneumonia/yes/' + yes[i] + '.dcm')
