import csv
import random

def create_entity_list():
    header = ["org_id", "remap_id"]
    new_rows = []
    with open("../Data/ml1m/joint-kg/kg/e_map.dat", 'r', encoding='latin-1') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            row[0], row[1] = row[1], row[0]
            new_rows.append(row)
    file.close()

    with open("../Data/ml1m/entity_list.txt", 'w+') as file:
        file.write(' '.join(header) + "\n")
        for row in new_rows:
            s = ' '.join(row)
            file.write(s)
            file.write("\n")
    file.close()

def create_item_list():
    header = ["org_id", "remap_id", "freebase_id"]
    file = open("../Data/ml1m/joint-kg/i2kg_map.tsv", "r")
    reader = csv.reader(file, delimiter="\t")
    dblink2_mlid = {}
    for row in reader:
        dblink2_mlid[row[2]] = int(row[0])
    file.close()

    file = open("../Data/ml1m/joint-kg/kg/e_map.dat", "r")
    reader = csv.reader(file, delimiter="\t")
    fileo = open("../Data/ml1m/item_list.txt", 'w+')
    writer = csv.writer(fileo, delimiter=" ")
    writer.writerow(header)
    for row in reader:
        dblink = row[1]
        if dblink not in dblink2_mlid: continue
        writer.writerow([dblink2_mlid[dblink], row[0], dblink])

    file.close()
    fileo.close()

def get_valid_products():
    valid_products = set()
    file = open("../Data/ml1m/item_list.txt", "r")
    reader = csv.reader(file, delimiter=" ")
    next(reader, None)
    for row in reader:
        ml_pid = int(row[0])
        valid_products.add(ml_pid)
    return valid_products

def review_train_test_split():
    uid_review_tuples = {}
    dataset_size = 0
    valid_products = get_valid_products()
    with open("../Data/ml1m/raw/ml-1m/ratings.dat", 'r', encoding='latin-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        for row in csv_reader:
            row = ''.join(row).strip().split("::")
            if int(row[1]) not in valid_products: continue
            if row[0] not in uid_review_tuples:
                uid_review_tuples[row[0]] = []
            uid_review_tuples[row[0]].append((row[0], row[1], row[2], row[3]))
            dataset_size += 1
    csv_file.close()

    for uid, reviews in uid_review_tuples.items():
        reviews.sort(key=lambda x: int(x[3]))

    train = []
    test = []
    invalid = 0
    th = 5
    for uid, reviews in uid_review_tuples.items(): #python dict are sorted, 1...nuser
        if len(reviews) < th:
            invalid += 1
            continue
        n_elements_test = int(len(reviews)*0.8)
        train.append(reviews[:n_elements_test])
        test.append(reviews[n_elements_test:])

    print("Removed {} users with less than {} reviews".format(invalid, th))
    with open("../Data/ml1m/train.txt", 'w+') as file:
        for uid, rec in enumerate(train): #rec = uid, item, rating, timestamp
            file.write(str(uid+1) + " ")
            s = ' '.join([review_tuple[1] for review_tuple in rec])
            file.writelines(s)
            file.write("\n")
    file.close()

    with open("../Data/ml1m/test.txt", 'w+') as file:
        for uid, rec in enumerate(test):
            file.write(str(uid+1) + " ")
            s = ' '.join([review_tuple[1] for review_tuple in rec])
            file.writelines(s)
            file.write("\n")
    file.close()


def get_itemid_mapping():
    pid_mapping = {}
    user_file = open("../Data/ml1m/item_list.txt", "r")
    reader = csv.reader(user_file, delimiter=" ")
    next(reader, None)
    for row in reader:
        old_id = int(row[0])
        new_id = int(row[1])
        pid_mapping[old_id] = new_id
    return pid_mapping


def create_kg_final():
    new_rows = []
    rmapping = {'0': '0', '1': '1', '8': '2', '3': '2', '10': '3', '14': '4' ,'15': '5', '16': '6', '18': '7', '2': '8'}
    with open("../Data/ml1m/joint-kg/kg/datasetraw.dat", 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            row[1], row[2] = row[2], row[1]
            row[1] = rmapping[row[1]]
            new_rows.append(row)
    file.close()

    with open("../Data/ml1m/kg_final.txt", 'w+') as file:
        for row in new_rows:
            s = ' '.join(row)
            file.write(s)
            file.write("\n")
    file.close()

def create_user_list():
    header = ["org_id", "remap_id"]
    new_rows = []
    with open("../Data/ml1m/raw/ml-1m/users.dat", 'r') as file:
        reader = csv.reader(file)
        for new_uid, row in enumerate(reader):
            row = row[0].split("::")
            new_rows.append([row[0], str(new_uid)])
    file.close()

    with open("../Data/ml1m/user_list.txt", 'w+') as file:
        file.write(' '.join(header) + "\n")
        for row in new_rows:
            s = ' '.join(row)
            file.write(s)
            file.write("\n")
    file.close()

def unify_dataset():
    kg_completion_path = "../Data/ml1m/joint-kg/kg/"
    selected_relations = ['0','1','8','3','10','14','15','16','18','2']
    print("Unifying dataset from joint-kg Knowledge graph completation for ML1M...")
    with open(kg_completion_path + "datasetraw.dat", 'w+', newline='\n') as dataset_file:
        print("Loading joint-kg train...")
        with open(kg_completion_path + "train.dat") as joint_kg_train:
            csv_reader = csv.reader(joint_kg_train, delimiter='\t')
            for row in csv_reader:
                relation = row[2]
                if relation not in selected_relations: continue
                dataset_file.writelines('\t'.join(row))
                dataset_file.write("\n")
        joint_kg_train.close()
        print("Loading joint-kg valid...")
        with open(kg_completion_path + "valid.dat") as joint_kg_valid:
            csv_reader = csv.reader(joint_kg_valid, delimiter='\t')
            for row in csv_reader:
                relation = row[2]
                if relation not in selected_relations: continue
                dataset_file.writelines('\t'.join(row))
                dataset_file.write("\n")
        joint_kg_valid.close()
        print("Loading joint-kg test...")
        with open(kg_completion_path + "test.dat") as joint_kg_test:
            csv_reader = csv.reader(joint_kg_test, delimiter='\t')
            for row in csv_reader:
                relation = row[2]
                if relation not in selected_relations: continue
                dataset_file.writelines('\t'.join(row))
                dataset_file.write("\n")
        joint_kg_test.close()
        print("Unifying dataset from joint-kg Knowledge graph completation... DONE")
    dataset_file.close()

def get_uid_mapping():
    file = open("../Data/lastfm/user_list.txt", 'r')
    reader = csv.reader(file, delimiter=" ")
    dataset_uid2kg_uid = {}
    next(reader, None)
    for row in reader:
        dataset_uid = int(row[0])
        kg_uid = int(row[1])
        dataset_uid2kg_uid[dataset_uid] = kg_uid
    file.close()
    return dataset_uid2kg_uid

def user_to_gender_map():
    dataset_uid2kg_uid = get_uid_mapping()
    file = open("../Data/lastfm/user2gender_map.txt", "w+")
    user_file = open("../Data/lastfm/users.dat")
    reader = csv.reader(user_file, delimiter=",")
    writer = csv.writer(file, delimiter="\t")
    writer.writerow(["uid", "gender"])
    next(reader, None)
    for row in reader:
        gender = row[3]
        uid = dataset_uid2kg_uid[int(row[0])]
        writer.writerow([uid, gender])
    file.close()
    user_file.close()

def remove_users_with_lown():
    file = open("../Data/lastfm/train.txt", 'r')
    reader = csv.reader(file, delimiter=" ")
    invalid_users = set()
    valid_train = {}
    valid_test = {}
    for row in reader:
        uid = int(row[0])
        train_list = []
        for c in range(1, len(row)):
            train_list.append(row[c])
        if len(train_list) < 4:
            invalid_users.add(uid)
            continue
        valid_train[uid] = train_list
    file.close()
    file = open("../Data/lastfm/train.txt", 'w+')
    writer = csv.writer(file, delimiter=" ")
    for u, list in valid_train.items():
        writer.writerow([u, *list])
    file.close()

    file = open("../Data/lastfm/test.txt", 'r')
    reader = csv.reader(file, delimiter=" ")
    for row in reader:
        uid = int(row[0])
        test_list = []
        for c in range(1, len(row)):
            test_list.append(row[c])
        if uid in invalid_users:
            continue
        valid_test[uid] = test_list
    file.close()

    file = open("../Data/lastfm/test.txt", 'w+')
    writer = csv.writer(file, delimiter=" ")
    for u, list in valid_test.items():
        writer.writerow([u, *list])
    file.close()

if __name__ == '__main__':
    #unify_dataset()
    #user_to_gender_map()
    remove_users_with_lown()
    exit(1)
    create_entity_list()
    create_item_list()
    create_user_list()
    create_kg_final()
    review_train_test_split()
