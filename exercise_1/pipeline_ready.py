import luigi
import csv


class ParseEmailsTask(luigi.Task):
    input_path = luigi.Parameter(default='exercise_1/db/registration_list.csv')
    output_path = luigi.Parameter(default='exercise_1/db/email_parsed.csv')

    def requires(self):
        return []

    def output(self):
        return [luigi.LocalTarget(self.output_path)]

    def run(self):
        with open(self.input_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            first_names = list()
            last_names = list()
            for row in csv_reader:
                rm_domain_name = row[0].split('@')[0]
                first_names.append(rm_domain_name.split('.')[0])
                last_names.append(rm_domain_name.split('.')[1])

            with open(self.output_path, 'a') as f:
                f.write('first_name, last_name\n')
                writer = csv.writer(f)
                writer.writerows(zip(first_names, last_names))


class CompareNamesTask(luigi.Task):
    input_path = luigi.Parameter(default='exercise_1/db/email_parsed.csv')
    output_path = luigi.Parameter(default='exercise_1/db/names_comparator.csv')

    def requires(self):
        return [ParseEmailsTask()]

    def output(self):
        return [luigi.LocalTarget(self.output_path)]

    def run(self):
        with open(self.input_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            first_names = list()
            is_first_name_longer = list()
            for row in csv_reader:
                first_names.append(row[0])
                if len(row[0]) > len(row[1]):
                    is_first_name_longer.append(True)
                else:
                    is_first_name_longer.append(False)

            with open(self.output_path, 'a') as f:
                f.write('first_name, is_first_name_longer\n')
                writer = csv.writer(f)
                writer.writerows(zip(first_names, is_first_name_longer))
