import glob
import tqdm


filenames = glob.glob('../dataset/*/*')
for fdir in tqdm.tqdm(filenames):
    with open(fdir, "r") as f:
        records = f.read()
    records = records.strip("\n").split("\n")
    try:
        for record in records:
            time, direction = record.strip("\n").split("\t")
            time = float(time)
            direction = float(direction)
    except Exception as e:
        print(fdir)
