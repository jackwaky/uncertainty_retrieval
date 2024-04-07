def read_file(file_path):
    pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            # print(line.strip().split(';')[:-1])
            # exit(0)
            pairs.append(tuple(line.strip().split(';')[:-1]))
    return pairs

def compare_files(file1_path, file2_path):
    file1_pairs = set(read_file(file1_path))
    file2_pairs = set(read_file(file2_path))

    print(len(file1_pairs), len(file2_pairs))

    common_pairs = file1_pairs.intersection(file2_pairs)

    return common_pairs

if __name__ == "__main__":
    domains = ['dress', 'shirt', 'toptee']
    
    for domain in domains:
        file1_path = f"/mnt/sting/jaehyun/dataset/fashionIQ/captions_pairs/fashion_iq-val-cap-{domain}.txt"
        file2_path = f"/mnt/sting/jaehyun/dataset/fashionIQ/captions_pairs/ambiguous_fashion_iq-val-cap-{domain}.txt"
    
        common_pairs = compare_files(file1_path, file2_path)
    
        if common_pairs:
            pass
        else:
            print("No common pairs found.")
