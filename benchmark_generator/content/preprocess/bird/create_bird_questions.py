import json
import re
import os

def write_jsonl(data: list[dict[str,str]], file_path: str):
    f_o_sql = open('bird_sql.jsonl', 'w')
    with open(file_path, "w", encoding="utf-8") as file:
        for offset, item in enumerate(data):
            file.write(json.dumps(item))
            file.write("\n")
            f_o_sql.write(str(offset + 1) + '.  ' + item['sql'] + '\n\n\n')
    f_o_sql.close()

def convert_to_jsonl(data: list, file_name: str):
    res = []
    db_table_sep = '-#-'
    for datum_idx, datum in enumerate(data):
        sql = datum["SQL"]
        res.append({
            "id": datum_idx,
            "db_id": datum["db_id"],
            "question": datum["question"],
            "sql": datum["SQL"]
        })
    write_jsonl(res, file_name)

def find_one(text, sub_str):
    pos_1 = text.find(sub_str)
    if pos_1 >= 0:
        pos_2 = text.find(sub_str, pos_1 + len(sub_str))
        if pos_2 >= 0:
            return -1
        else:
            return pos_1
    return -1

def check_simple(sql):
    if ' JOIN ' in sql:
        return False
    from_text = ' FROM '
    from_pos = find_one(sql, from_text)
    if from_pos < 0:
        return False
    pos_2 = find_one(sql, ' WHERE ')
    if pos_2 < 0:
        return False
    table_text = sql[from_pos + len(from_text):pos_2]
    if len(table_text.split(',')) > 1:
        print(table_text)
    return True

def main():
    out_file = 'bird_questions.jsonl'
    if os.path.isfile(out_file):
        print(f'{out_file} already exists')
        return
    with open("/home/cc/text2sql_bird/dev_20240627/dev.json") as f:
        dev = json.load(f)
    with open("/home/cc/text2sql_bird/train/train.json") as f:
        train = json.load(f)
    
    all_data = train + dev
    filtered_all = [a for a in all_data if check_simple(a['SQL'])]
    convert_to_jsonl(filtered_all, out_file)

if __name__ == '__main__':
    main()
