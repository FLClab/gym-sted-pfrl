import bz2
import pickle
import numpy as np
from matplotlib import pyplot as plt
from gym_sted import defaults

import metrics
from skimage.feature import peak_local_max

subdir_names = np.arange(1024, 100000, 1024).tolist()
subdir_names.append(100000)
for idx, num in enumerate(subdir_names):
    if num != 100000:
        subdir_names[idx] = f"{num}_checkpoint/"
    else:
        subdir_names[idx] = f"{num}_finish/"

# read the pbz2 dict, then what?

# main_path = "./data/exp_easy_env_training_v03_suite_2/20210920-092334_6c34c75d/"
main_path = "./data/exp_hard_env_training_v03_suite_2/20210920-092334_e272790e/"

for subdir in subdir_names:

    print(f"!!!!!!!")
    print(subdir)
    print(f"!!!!!!!")

    path_to_dir = main_path + subdir
    f = bz2.BZ2File(path_to_dir + "lighter_episodes_dict.pbz2")
    episodes_dict = pickle.load(f)
    f.close()

    # the first analysis idea is to figure out if an episode is successful or not (have >=80% of NDs been id'd ?)
    fixed_n_nanodomains_threshold = 0.8
    episode_success_dict = {}

    for episode_key in episodes_dict:
        episode_success_dict[episode_key] = {}


        sted_images = episodes_dict[episode_key]["per_step"]["sted_image_s"]
        pdt_s = episodes_dict[episode_key]["per_step"]["pdt_s"]
        p_ex_s = episodes_dict[episode_key]["per_step"]["p_ex_s"]
        p_sted_s = episodes_dict[episode_key]["per_step"]["p_sted_s"]
        nd_positions = episodes_dict[episode_key]["episodic"]["ND_positions"]
        nd_assigned_truth_list = []
        for i in range(nd_positions.shape[0]):
            nd_assigned_truth_list.append(0)
        new_threshold = np.floor(fixed_n_nanodomains_threshold *
                                 nd_positions.shape[0]) / nd_positions.shape[0]
        episode_success_dict[episode_key]["adjusted_th"] = new_threshold
        # print(f"nb NDs = {nd_positions.shape[0]}")
        # print(f"80% of {nd_positions.shape[0]} is {0.8 * nd_positions.shape[0]}")
        # print(f"adjusted threshold is {new_threshold}")
        # print(f"{int(100 * new_threshold)}% of {nd_positions.shape[0]} is {new_threshold * nd_positions.shape[0]}")

        guess_coords_list = []

        for t in range(len(sted_images)):
            guess_coords = peak_local_max(sted_images[t], min_distance=2, threshold_rel=0.5)
            guess_coords_list.append(guess_coords)

            detector = metrics.CentroidDetectionError(nd_positions, guess_coords, 2, algorithm="hungarian")
            for nd in detector.truth_couple:
                nd_assigned_truth_list[nd] = 1

            # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            # ax.imshow(sted_images[t])
            # ax.scatter(nd_positions[:, 1], nd_positions[:, 0], marker="x", color="k", label="GTs")
            # ax.scatter(guess_coords[:, 1], guess_coords[:, 0], color="r", label="Guesses")
            # fig.suptitle(f"t = {t + 1} / {len(sted_images)}, \n"
            #              f"pdt = {round(100 * pdt_s[t] / defaults.action_spaces['pdt']['high'], 2)} %, \n"
            #              f"p_ex = {round(100 * p_ex_s[t] / defaults.action_spaces['p_ex']['high'], 2)} %, \n"
            #              f"p_sted = {round(100 * p_sted_s[t] / defaults.action_spaces['p_sted']['high'], 2)} %, \n")
            # plt.show()

        episode_success_dict[episode_key]["nd_assigned_list"] = nd_assigned_truth_list
        episode_success_dict[episode_key]["id_ratio"] = np.sum(nd_assigned_truth_list) / len(nd_assigned_truth_list)

    n_successful = 0
    n_episodes = 0
    successful_episodes = []
    for key in episode_success_dict:
        if episode_success_dict[key]["id_ratio"] >= episode_success_dict[episode_key]["adjusted_th"]:
            n_successful += 1
            successful_episodes.append(key)
        n_episodes += 1

    # write to a file the number of episodes, number of successful episodes, %
    # save a numpy array of the indices of the successful episodes
    f = open(path_to_dir + "analysis_episodes_stats.txt", "w")
    f.write(f"{n_episodes}\n")
    f.write(f"{n_successful}\n")
    f.write(f"{n_successful / n_episodes}")
    f.close()
    successful_episodes = np.array(successful_episodes)
    np.save(path_to_dir + "successful_episodes_idx.npy", successful_episodes)

    print(f"{n_successful} successful episodes out of {n_episodes} ({round(100 * n_successful / n_episodes, 2)}%)")
    print(f"succesful episode idx : {successful_episodes}")

# a next analysis idea would be to split the exp in "before the flash", "during the flash", "after the flash" sections
# and look at the selected actions in each section
