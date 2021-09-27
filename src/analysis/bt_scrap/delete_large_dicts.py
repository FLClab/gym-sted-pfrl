import os

main_dirs = ["./data/exp_easy_env_training_v03/20210912-142004_32a8131e/",
             "./data/exp_hard_env_training_v03/20210912-142004_f448f0e1/",
             "./data/exp_easy_env_training_v03_suite_1/20210915-090205_e0091d19/"
             "./data/exp_hard_env_training_v03_suite_1/20210915-090223_b7ff6691/"]

for main_dir in main_dirs:
    agent_subdirs = []
    for subdir, _, _ in os.walk(main_dir):
        split_subdir = subdir.split("/")
        dir_name_split = split_subdir[-1].split("_")
        if dir_name_split[0] == "best" or dir_name_split[-1] == "checkpoint" or dir_name_split[-1] == "finish":
            agent_subdirs.append(split_subdir[-1])

    for data_dir in agent_subdirs:
        if os.path.isfile(main_dir + data_dir + "/lighter_episodes_dict.pbz2"):
            os.remove(main_dir + data_dir + "/lighter_episodes_dict.pbz2")




