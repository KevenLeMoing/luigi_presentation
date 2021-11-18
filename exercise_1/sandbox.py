import csv


with open('exercise_1/db/registration_list.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    first_names = list()
    last_names = list()
    for row in csv_reader:
        rm_domain_name = row[0].split('@')[0]
        first_names.append(rm_domain_name.split('.')[0])
        last_names.append(rm_domain_name.split('.')[1])

    with open('exercise_1/db/output.csv', 'a') as f:
        f.write('first_name, last_name\n')
        writer = csv.writer(f)
        writer.writerows(zip(first_names, last_names))
