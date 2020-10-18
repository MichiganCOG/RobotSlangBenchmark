''' Plotting losses etc.  '''


import json
import pandas as pd
import os
from scipy.stats import ttest_rel
import numpy as np
from rslang_simulator import RobotSlangSimulator
import tqdm


def _score_item(data, metric):
    ''' Calculate error based on the final position in trajectory, and also 
        the closest position (oracle stopping rule). '''
    # shortest path (load it one at a time)
    path = data['trajectory']
    try:
        env = RobotSlangSimulator(data['scan'])
    except TypeError:  # DEBUG
        print(data['scan'])  # DEBUG
        return 0  # DEBUG
    planner = env.planner
    # Planner to get topological distanced

    start = env.agent.pose[:,:2][0]  
    goal  = env.goal.pose[:,:2][0]
    final_position = path[-1]

    # Distance from final position to goal
    if metric == 'top_final_goal':
        value = planner.get_top_dist(final_position, goal)
    elif metric == 'oracle_top_final_goal':
        value = np.min([planner.get_top_dist(p, goal) for p in path])
    elif metric == 'success':
        dist = planner.get_top_dist(final_position, goal)
        value = 1 if dist < 2 * 0.254 else 0  # [ https://arxiv.org/pdf/1807.06757.pdf ]
    elif metric == 'spl':
        dist = planner.get_top_dist(final_position, goal)
        success = 1 if dist < 2 * 0.254 else 0
        best_path_dist = planner.get_top_dist(start, goal)
        value = success * best_path_dist / max(best_path_dist, dist)
    else:
        raise(ValueError, 'Unrecognized metric "%s"' % metric)

    # Store the metrics
    return value

PLOT_DIR = 'FINAL_RESULTS/plots/'
RESULTS_DIR = 'FINAL_RESULTS/results/'

max_iter = 2000

dfs = {}
# val-seq2seq-all-planner_path-sample-imagenet-log
inf_inst_idxs = set()
seed_avg_metric_by_traj = None
n_seeds = 3
for seed in range(n_seeds):
    print("COLLATING FOR SEED %d" % seed)

    summary = {"val": {}, "test": {}}
    metric_by_traj = {"val": {}, "test": {}}
    for path_type, path_len in [['trusted_path', 120]]:
        print(path_type)
        for eval_type in ['val', 'test']:
            print('\t%s (%d)' % (eval_type, path_len))
            for feedback in ['sample', 'teacher']:
                lr = '001000' if feedback == 'sample' else '000100'
                print('\t\t%s' % feedback)
                for mod in ['blind-', 'blind-language', 'blind-vision']:
                    print('\t\t\t%s' % mod)
                    # TODO: process seeds [0, 1, 2] not just seed 0; just getting debugging done now.
                    fn_dir = '%s_feedback-%s_%s_lr-0.%s_hs-64_we-64/seed-%d/' % (eval_type, feedback, mod, lr, seed)
                    fn = None
                    for _, _, fns in os.walk(os.path.join(PLOT_DIR, fn_dir)):
                        if len(fns) == 0:
                            continue
                        for cfn in fns:
                            if 'hs_64-we_64' in cfn:
                                base_fn = cfn
                                fn = os.path.join(PLOT_DIR, fn_dir, base_fn)
                                break
                        if fn is None and len(fns) > 1:
                            print("WARNING multi-log with no heuristic match to 'hs_64-we_64'", fn_dir, fns)
                    if fn is not None and os.path.isfile(fn):
                        log = '%s-%s-%s' % (eval_type, feedback, mod)
                        dfs[log] = pd.read_csv(fn)
                        print('\t\t\t\t%d' % len(dfs[log]))
                        metrics = [
                            # 'val_seen success_rate',
                            # 'val_seen oracle path_success_rate',
                            # 'val_seen dist_to_end_reduction',
                            # 'val_unseen success_rate',
                            # 'val_unseen oracle path_success_rate',
                            'val_seen top_final_goal', 'val_seen top_oracle_goal'] if eval_type == 'val' else [
                            # 'test success_rate',
                            # 'test oracle path_success_rate',
                            'test top_final_goal', 'test top_oracle_goal']
                        for metric in metrics:
                            best_row = dfs[log].loc[dfs[log][metric].idxmin()]
                            v = best_row[metric]
                            iteration = best_row['iteration']
                            if max_iter is not None:
                                iteration = min(max_iter, iteration)
                            print('\t\t\t\t%s\t%.3f\t(%d)' % (metric, v, iteration))

                        # Populate summary.
                        abl = '%s-%s' % (feedback, mod)
                        if abl not in summary[eval_type]:
                            summary[eval_type][abl] = {}
                            metric_by_traj[eval_type][abl] = {}
                            for m in ['top_final_goal', 'top_oracle_goal']:
                                # summary[cond][abl] = {"if": {}, "gd": {}}
                                summary[eval_type][abl][m] = {}
                                metric_by_traj[eval_type][abl][m] = {}
                        # ifm = '%s oracle path_success_rate' % cond
                        # if ifm in dfs[log]:
                        #     summary[cond][abl]["if"][path_type] = list(dfs[log][ifm])
                        cond = eval_type + "_seen" if eval_type == 'val' else eval_type
                        for gdm in ['%s top_final_goal' % cond, '%s top_oracle_goal' % cond,
                                    '%s success' % cond, '%s spl' % cond]:
                            m = gdm.split()[-1]
                            if gdm in dfs[log]:
                                summary[eval_type][abl][m][path_type] = list(dfs[log][gdm])

                            # Find and read associated data from results JSON.
                            # If not test, just get best performance across this fold.
                            # Perf is always at the epoch that was best for the main metric, top_final_goal.
                            if eval_type != "test":
                                best_row = dfs[log].loc[dfs[log]['%s top_final_goal' % gdm.split()[0]].idxmin()]
                                iteration = best_row['iteration']
                                if max_iter is not None:
                                    iteration = min(max_iter, iteration)
                                # print(log, iteration)  # DEBUG
                                # _ = input()  # DEBUG

                            else:  # get test perf at best iter from val_seen (there's no 'unseen' val in RoboSlang)
                                val_log = '%s-%s-%s' % ("val", feedback, mod)
                                best_row = dfs[val_log].loc[dfs[val_log]['val_seen top_final_goal'].idxmin()]
                                iteration = best_row['iteration']
                                if max_iter is not None:
                                    iteration = min(max_iter, iteration)
                                # print(log, iteration)  # DEBUG
                                # _ = input()  # DEBUG
                                # test-seq2seq-all-trusted_path-80-sample-imagenet 6200.0

                            # Open the json of individual trajectory scores at this iteration.
                            rfn = os.path.join(RESULTS_DIR, fn_dir,
                                               "%s_%s_iter_%d.json" % (base_fn.split('-log')[0], cond, iteration))
                            if not os.path.isfile(rfn):  # Should go away after full rerun.
                                print("WARNING: '%s' does not exist, but iter is best" % rfn)
                                found = False
                                iter_sub = 1
                                while not found:
                                    rfn = os.path.join(RESULTS_DIR, fn_dir,
                                                   "%s_%s_iter_%d.json" % (base_fn.split('-log')[0], cond, iteration - iter_sub))
                                    if os.path.isfile(rfn):
                                        found = True
                                    iter_sub += 1
                                print("... using '%s' instead" % rfn)
                            with open(rfn, 'r') as f:
                                rd = json.load(f)
                            if m not in rd[0]:
                                print("Calculating missing '%s' data..." % m)
                                for idx in tqdm.tqdm(range(len(rd))):
                                    rd[idx][m] = _score_item(rd[idx], m)
                                print("... done; caching data back to file...")
                                with open(rfn, 'w') as f:
                                    json.dump(rd, f)
                                print("... done; wrote back to '%s'" % rfn)
                            metric_by_traj[eval_type][abl][m][path_type] = \
                                {rd[idx]["inst_idx"]: rd[idx][m]
                                 for idx in range(len(rd))}
                            for inst_idx in metric_by_traj[eval_type][abl][m][path_type]:
                                if not np.isfinite(metric_by_traj[eval_type][abl][m][path_type][inst_idx]):
                                    inf_inst_idxs.add(inst_idx)
                    else:
                        print("WARNING: %s Missing file log in dir '%s'" % (eval_type, os.path.join(PLOT_DIR, fn_dir)))

    # Across seeds, perform a micro-average at the trajectory level to get average performance.
    # If this is first run, copy over structure and then norm by n_seeds
    if seed_avg_metric_by_traj is None:
        seed_avg_metric_by_traj = metric_by_traj
        for eval_type in seed_avg_metric_by_traj:
            for abl in seed_avg_metric_by_traj[eval_type]:
                for m in ['top_final_goal', 'top_oracle_goal']:
                    for path_type in seed_avg_metric_by_traj[eval_type][abl][m]:
                        for inst_idx in seed_avg_metric_by_traj[eval_type][abl][m][path_type]:
                            seed_avg_metric_by_traj[eval_type][abl][m][path_type][inst_idx] = \
                                [seed_avg_metric_by_traj[eval_type][abl][m][path_type][inst_idx]]
    # If this is a subsequent run, add values normed by n_seeds
    else:
        for eval_type in seed_avg_metric_by_traj:
            for abl in seed_avg_metric_by_traj[eval_type]:
                for m in ['top_final_goal', 'top_oracle_goal']:
                    for path_type in seed_avg_metric_by_traj[eval_type][abl][m]:
                        for inst_idx in seed_avg_metric_by_traj[eval_type][abl][m][path_type]:
                            seed_avg_metric_by_traj[eval_type][abl][m][path_type][inst_idx].append(
                                metric_by_traj[eval_type][abl][m][path_type][inst_idx])

if len(inf_inst_idxs) > 0:
    print("WARNING: there are %d inf_inst_idxs entries" % len(inf_inst_idxs))

# Show seed average results, then average values.
print("Averages and std over seeds")
for eval_type in seed_avg_metric_by_traj:
    print('\t%s' % eval_type)
    for abl in seed_avg_metric_by_traj[eval_type]:
        print('\t\t%s' % abl)
        for m in ['top_final_goal', 'top_oracle_goal']:
            for path_type in seed_avg_metric_by_traj[eval_type][abl][m]:
                print('\t\t\t%s\t%s\t %.2f\t(%.2f)' %
                      (path_type, m,
                       np.average([np.average(seed_avg_metric_by_traj[eval_type][abl][m][path_type][inst_idx])
                                   for inst_idx in seed_avg_metric_by_traj[eval_type][abl][m][path_type]]),
                       np.std([np.average(seed_avg_metric_by_traj[eval_type][abl][m][path_type][inst_idx])
                               for inst_idx in seed_avg_metric_by_traj[eval_type][abl][m][path_type]])))
                for inst_idx in seed_avg_metric_by_traj[eval_type][abl][m][path_type]:
                    seed_avg_metric_by_traj[eval_type][abl][m][path_type][inst_idx] = np.average(
                        seed_avg_metric_by_traj[eval_type][abl][m][path_type][inst_idx])

# Run statistical significance tests.
alpha = 0.05
k = None
sig_values = set()
p_values = []  # list of tuples (p value, test condition 1, test condition 2)
comp_abls = [[('teacher', 'blind-'), ('sample', 'blind-')],
             [('teacher', 'blind-'), ('teacher', 'blind-language')],
             [('teacher', 'blind-'), ('teacher', 'blind-vision')],
             [('sample', 'blind-'), ('sample', 'blind-language')],
             [('sample', 'blind-'), ('sample', 'blind-vision')]]
sups = ['trusted_path']
m = 'top_final_goal'
for cond in ['val', 'test']:
    print(cond)
    for all_abls in comp_abls:
        for hb_idx in range(len(all_abls) - 1):
            feedback, mod = all_abls[hb_idx]
            abl = '%s-%s' % (feedback, mod)
            if abl not in seed_avg_metric_by_traj[cond]:
                continue
            print('\t%s' % abl)
            for hb_jdx in range(hb_idx + 1, len(all_abls)):
                _feedback, _mod = all_abls[hb_jdx]
                _abl = '%s-%s' % (_feedback, _mod)
                if _abl not in seed_avg_metric_by_traj[cond]:
                    continue
                print('\t\t%s' % _abl)
                for sup in ['trusted_path']:
                    if (sup not in seed_avg_metric_by_traj[cond][abl][m] or
                            sup not in seed_avg_metric_by_traj[cond][_abl][m]):
                        continue
                    inst_idxs = set(seed_avg_metric_by_traj[cond][abl][m][sup].keys()).union(
                        set(seed_avg_metric_by_traj[cond][_abl][m][sup].keys()))
                    a = [seed_avg_metric_by_traj[cond][abl][m][sup][inst_idx] for inst_idx in inst_idxs
                         if inst_idx not in inf_inst_idxs]
                    b = [seed_avg_metric_by_traj[cond][_abl][m][sup][inst_idx] for inst_idx in inst_idxs
                         if inst_idx not in inf_inst_idxs]
                    if len(seed_avg_metric_by_traj[cond][abl][m][sup]) - len(a) > 0:
                        print("\t\t\tWARNING: removed %d inf entries from a" %
                              (len(seed_avg_metric_by_traj[cond][abl][m][sup]) - len(a)))
                    if len(seed_avg_metric_by_traj[cond][_abl][m][sup]) - len(b) > 0:
                        print("\t\t\tWARNING: removed %d inf entries from b" %
                              (len(seed_avg_metric_by_traj[cond][_abl][m][sup]) - len(b)))
                    # We can run a paired test because (a,b) represent samples from the same trajectories under two evals.
                    _, p = ttest_rel(a, b)
                    if not np.isnan(p):
                        p_values.append((p, (cond, abl, sup), (cond, _abl, sup)))
                    else:
                        print("\t\t\tp is nan!")
                        print("a", a)
                        print("b", b)
                    print("\t\t\t%s\tN=%d\t(%.2f, %.2f)\tp=%.3f" % (sup, len(a), np.average(a), np.average(b), p))

# Run the Benjamini–Yekutieli procedure to control FDR and report significance.
# https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Yekutieli_procedure
if k is None:
    print("# tests", len(p_values))  # DEBUG
    sorted_p_values = sorted(p_values, key=lambda kv: kv[0])
    m = len(sorted_p_values)
    cm = sum([1. / i for i in range(1, m + 1)])
    for idx in range(len(sorted_p_values)):
        print(idx, sorted_p_values[idx][1], sorted_p_values[idx][2], sorted_p_values[idx][0], ((idx + 1) * alpha) / (m * cm))  # DEBUG
        if sorted_p_values[idx][0] <= ((idx + 1) * alpha) / (m * cm):
            k = idx + 1
            sig_values.add((sorted_p_values[idx][1], sorted_p_values[idx][2]))
        else:
            break
    print("alpha", alpha, "m", m, "c(cm)", cm, "k", k)
print('\n'.join([str(sv) for sv in sig_values]))

