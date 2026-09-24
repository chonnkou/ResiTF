import math
import numpy as np
import random
import hypernetx as hnx
import warnings 
warnings.simplefilter('ignore')
import copy
import matplotlib.pyplot as plt
import json
import os
import time
from datetime import datetime
from itertools import combinations
from joblib import Parallel, delayed
from collections import Counter
from pathlib import Path
import signal
from scipy.sparse.linalg import eigs
# Search-method baselines used by the comparison helpers below.
import simulated_annealing as sa
import simulated_annealing_mu2 as mu2
import greedy_repair as gr

def algebraic_connectivity(X):
	"""
	Calculate the algebraic connectivity of a binary hypergraph
	H : numpy.ndarray
		Incidence matrix with shape (N, K).
		Rows = nodes (agents)
		Columns = hyperedges (tasks)
		H[i, k] = 1 if node i belongs to hyperedge k.
	Returns
	mu2 : float
		Second-smallest eigenvalue of the EDVW Laplacian.
	"""

	X = np.asarray(X, dtype=float)
	N, T = X.shape
	
	# incidence weight or a.k.a. EDVW
	gamma = X.copy()
	# hyperedge weight
	omega = np.sum(X, axis=0)

	# probability transition matrix of the random walk

	# incidence info and node weight
	R = X * gamma
	# incidence info and hyperedge weight
	W = X * omega
	
	# node degs - hyperedge weight W - sum over row - axis=1
	delta = np.sum(W, axis=1)
	# hyperedge degs - node weight R - sum over col - axis=0
	d = np.sum(R, axis=0)
	# make it float
	delta = np.maximum(delta, np.ones(N))
	d = np.maximum(d, np.ones(T))
	# probability transition matrix
	P = (
		np.diag(delta ** -1)
		@ W
		@ np.diag(d ** -1)
		@ R.T
	)

	P = np.nan_to_num(P)

	# construct Pi
	eigenValues, eigenVectors = eigs(
		P.T,
		k=1,
		which='LR'
	)
	pi = np.abs(eigenVectors[:, 0])
	Pi = np.diag(pi)

	# Laplacian
	L = Pi - (Pi @ P + P.T @ Pi) / 2.0

	# Remove tiny numerical asymmetry / imaginary parts
	L = np.real(L)
	L = (L + L.T) / 2.0

	# algebraic connectivity = second-smallest eigenvalue

	eigenvalues = np.linalg.eigvalsh(L)
	eigenvalues = np.sort(eigenvalues)

	mu2 = eigenvalues[1]

	if np.isclose(mu2, 0.0, atol=1e-12):
		mu2 = 0.0

	return mu2

def H2X(H):
	M_incidence = H.incidence_matrix()
	X = M_incidence.toarray()
	return np.asarray(X, dtype=int)

def X2H(X):
	return hnx.Hypergraph.from_incidence_matrix(X)

def retrive_H_aps(p):
	path = Path(p)
	data = []
	for line in path.read_text().splitlines():
		data.append([float(x) for x in line.split()])
	n_authors = len(data)
	n_papers = len(data[0])
	hyperedges = []
	for j in range(n_papers):
		hyperedges.append([i for i in range(n_authors) if data[i][j] != 0])
	return hnx.Hypergraph(hyperedges)

def update_e2n(e2n, n, es, B, n2d):
	resi_b = B - n2d[n]
	if resi_b <= 0:
		return e2n
	other_es = [e for e in e2n.keys() if e not in es]
	e2s = {e: len(e2n[e]) for e in other_es}
	select_es = sorted(e2s, key=e2s.get)[:resi_b]
#	select_es = random.sample(other_es, resi_b)
	for e in select_es:
		e2n[e].append(n)
	return e2n
	
def update_e2n_drop2(e2n, n, es, B, n2d):
	resi_b = B - n2d[n]
	if resi_b <= 0:
		return e2n
	other_es = [e for e in e2n.keys() if e not in es]
	select_es = random.sample(other_es, resi_b)
	for e in select_es:
		e2n[e].append(n)
	return e2n
	
	
def gene_fcr_drop2(N, T, B, R):
	# ablation experiment - drop prescriptive rule at stage 2
	# return H_drop2
	t_s = time.time()
	if N*B < T*R:
		return 'Error: N*B < T*R'
	indicator = (N*B) / T - R
	H = gene_fc(N, T, B, R)
	if indicator == 0:
		return H
	n2e = H.dual().incidence_dict
	e2n = H.incidence_dict
	n2d = {n: H.degree(n, s=1) for n in H.nodes}
	if indicator > 0:
		for n, es in n2e.items():
			e2n = update_e2n_drop2(e2n, n, es, B, n2d)
	return hnx.Hypergraph(e2n)
	
def gene_fcr_drop12(N, T, B, R):
	# ablation experiment - drop prescriptive rule at stage 1 and 2
	# return H_drop12
	t_s = time.time()
	if N*B < T*R:
		return 'Error: N*B < T*R'
	indicator = (N*B) / T - R
	H = gene_fc_drop1(N, T, B, R)
	if indicator == 0:
		return H
	n2e = H.dual().incidence_dict
	e2n = H.incidence_dict
	n2d = {n: H.degree(n, s=1) for n in H.nodes}
	if indicator > 0:
		for n, es in n2e.items():
			e2n = update_e2n_drop2(e2n, n, es, B, n2d)
			H_ = hnx.Hypergraph(e2n)
			meand = mean_degree(H_)
			if meand >= B:
				break
	return H_
	
	
def mean_edges_size(H):
	e2n = H.incidence_dict
	sizes = [len(ns) for ns in e2n.values()]
	return math.ceil(sum(sizes) / len(sizes))

def gene_fcr(N, T, B, R):
	# use total additional budget to build resilience
	t_s = time.time()
	if N*B < T*R:
		return 'Error: N*B < T*R'
	indicator = (N*B) / T - R
	H = gene_fc(N, T, B, R)
	if indicator == 0:
		return H
	n2e = H.dual().incidence_dict
	e2n = H.incidence_dict
	n2d = {n: H.degree(n, s=1) for n in H.nodes}
	if indicator > 0:
		for n, es in n2e.items():
			e2n = update_e2n(e2n, n, es, B, n2d)
	return hnx.Hypergraph(e2n)
	
def gene_fc(N, T, B, R):
	# return feasible and connected team
	if N*B < T*R:
		return 'Error: N*B < T*R'
	hyperedges = []
	selected = []
	not_selected = [i for i in range(N)]
	for i in range(T):
		if len(not_selected) >= R-1:
			if i == 0:
				current_e = random.sample(not_selected, R)
			else:
				current_e = random.sample(not_selected, R-1)
				current_e.extend(connector)

		elif len(not_selected) == 0:
			# assign ones with least degrees (highest residual budgets) to tasks
			current_e = find_agents(H, R)
		else:
			current_e = [n for n in not_selected]
			rest = find_agents(H, R-len(current_e))
			current_e.extend(rest)
			
		selected = list(set(selected + current_e))
		not_selected = [n for n in range(N) if n not in selected]
		hyperedges.append(current_e)
		H = hnx.Hypergraph(hyperedges)
		# select one with least degree as the connector
		connector = [find_connector(H)]

	return H
	
def gene_fcr_drop1(N, T, B, R):
	# ablation experiment - drop prescriptive rule at stage 1
	# return H_drop1
	t_s = time.time()
	if N*B < T*R:
		return 'Error: N*B < T*R'
	indicator = (N*B) / T - R
	H = gene_fc_drop1(N, T, B, R)
	if indicator == 0:
		return H
	n2e = H.dual().incidence_dict
	e2n = H.incidence_dict
	n2d = {n: H.degree(n, s=1) for n in H.nodes}
	if indicator > 0:
		for n, es in n2e.items():
			e2n = update_e2n(e2n, n, es, B, n2d)
			H_ = hnx.Hypergraph(e2n)
			meand = mean_degree(H_)
			if meand >= B:
				break
	return H_
	
def gene_fc_drop1(N, T, B, R):
	
	if N*B < T*R:
		return 'Error: N*B < T*R'
	hyperedges = []
	selected = []
	not_selected = [i for i in range(N)]
	for i in range(T):
		if len(not_selected) >= R-1:
			if i == 0:
				current_e = random.sample(not_selected, R)
			else:
				current_e = random.sample(not_selected, R-1)
				current_e.extend(connector)

		elif len(not_selected) == 0:
			# assign ones from selected at random
			current_e = random.sample(selected, R)
		else:
			current_e = [n for n in not_selected]
			rest = random.sample(selected, R-len(current_e))
			current_e.extend(rest)
			
		selected = list(set(selected + current_e))
		not_selected = [n for n in range(N) if n not in selected]
		hyperedges.append(current_e)
		H = hnx.Hypergraph(hyperedges)
		# select one from selected at random
		connector = random.sample(selected, 1)

	return H
	
def max_degree(H):
	degrees = [H.degree(n, s=1) for n in H.nodes]
	return max(degrees)
	
def min_degree(H):
	degrees = [H.degree(n, s=1) for n in H.nodes]
	return min(degrees)
	
def mean_degree(H):
	degrees = [H.degree(n, s=1) for n in H.nodes]
	return sum(degrees) / len(degrees)
	
def deg_info(H):
	return {'max': max_degree(H), 'min': min_degree(H), 'mean': mean_degree(H)}

def edges_size_info(H):
	e2n = H.incidence_dict
	sizes = [len(ns) for ns in e2n.values()]
	return {'max': max(sizes), 'min': min(sizes), 'mean': sum(sizes) / len(sizes)}
	
def H_deg_edge_info(H):
	info = {}
	info['deg'] = deg_info(H)
	info['edge'] = edges_size_info(H)
	return info

def component_num(H):
	coms = [comp for comp in H.s_components(s=1)]
	return len(coms)
	
def edge_size_distri(H):
	e2n = H.incidence_dict
	sizes = [len(ns) for ns in e2n.values()]
	return Counter(sizes)

def find_connector(H):
	n2d = {n: H.degree(n, s=1) for n in H.nodes}
	return min(n2d, key=n2d.get)

def find_agents(H, R):
	n2d = {n: H.degree(n, s=1) for n in H.nodes}
	return sorted(n2d, key=n2d.get)[:R]

def generate_interval(et):
	stop, num = et
	return list(np.arange(0, stop + stop / num, stop / num))


def saving_ana12(results, p_interval, runstep):
	save_dir = f"results_ana12"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"p_interval": p_interval,
		"runstep": runstep
	}
	filename = f"{stop}_{num}_{runstep}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")


def ana_12(H, p_interval, runstep):
	ps = generate_interval(p_interval)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	results = Parallel(n_jobs=8)(delayed(cc_num)(H, p) for p, i in tasks)
	
	saving_ana12(results, p_interval, runstep)

def update_ns(ns, p):
	ns_new = [n for n in ns if random.random() > p]
	if not ns_new:
		ns_new = random.sample(ns, 1)
	return ns_new


def after_failures(e2n, p):
	e2n_new = {}
	for e, ns in e2n.items():
		e2n_new[e] = update_ns(ns, p)
	return e2n_new

def cc_num(H, p):
	e2n = H.incidence_dict
	e2n_new = after_failures(e2n, p)
	H_new = hnx.Hypergraph(e2n_new)
	return component_num(H_new)


def ana_11(H, R, p_interval, runstep):
	ps = generate_interval(p_interval)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	results = Parallel(n_jobs=8)(delayed(unfeasi_num)(H, R, p) for p, i in tasks)
	
	saving_ana11(results, p_interval, runstep)
	
	
def saving_ana11(results, p_interval, runstep):
	save_dir = f"results_ana11"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"p_interval": p_interval,
		"runstep": runstep
	}
	filename = f"{stop}_{num}_{runstep}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
	
def unfeasi_num(H, R, p):
	e2n = H.incidence_dict
	e2n_new = after_failures(e2n, p)
	return len([1 for ns in e2n_new.values() if len(ns) < R])
	
def unfeasi_num_aps(H, R, p):
	e2n = H.incidence_dict
	e2n_new = after_failures(e2n, p)
	return len([1 for e, ns in e2n_new.items() if len(ns) < len(e2n[e])])
	
	
def saving_ana2(results, p_interval, runstep, B, R, H):
	problem_info = H.shape + (B, R)
	save_dir = f"results_ana2"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"p_interval": p_interval,
		"runstep": runstep
	}
	filename = f"{problem_info}_{stop}_{num}_{runstep}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1] + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def ana_2(H, B, R, p_interval, runstep):
	ps = generate_interval(p_interval)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	results = Parallel(n_jobs=8)(delayed(recovery_dis)(H, B, R, p) for p, i in tasks)
	
	saving_ana2(results, p_interval, runstep, B, R, H)
	
def distance_shortest(H, n, e, e2n):
	return min([H.distance(n, n_) for n_ in e2n[e]])
	
def recovery_dis(H, B, R, p):
	diameter = H.diameter()
	e2n = H.incidence_dict
	n2b = {n: H.degree(n, s=1) for n in H.nodes}
	e2n_new = after_failures(e2n, p)
	H_new = hnx.Hypergraph(e2n_new)
	n2d = {n: H_new.degree(n, s=1) for n in H_new.nodes}
	failed_es2need = {e: len(e2n[e]) - len(ns) for e, ns in e2n_new.items() if len(ns) < len(e2n[e])}
	patching_agents2supply = {n: n2b[n] - d for n, d in n2d.items() if d < n2b[n]}
	# patching
	patching_cost = 0
	penalty_cost = 0
	if not failed_es2need:
		return patching_cost + penalty_cost
	for fe, need in failed_es2need.items():
		pa2dis = {pa: distance_shortest(H_new, pa, fe, e2n_new) for pa in patching_agents2supply.keys()}
		for pa, dis in dict(sorted(pa2dis.items(), key=lambda x: x[1])).items():
			if dis > diameter:
				penalty_cost += diameter*2
				break
			if patching_agents2supply[pa] == need:
				patching_cost += dis
				del patching_agents2supply[pa]
				break
			if patching_agents2supply[pa] > need:
				patching_cost += dis
				patching_agents2supply[pa] = patching_agents2supply[pa] - need
				break
			if patching_agents2supply[pa] < need:
				patching_cost += dis
				need = need - patching_agents2supply[pa]
				del patching_agents2supply[pa]
				
	
	return patching_cost + penalty_cost
	
def recovery_dis_(H, B, R, p):
	diameter = H.diameter()
	e2n = H.incidence_dict
	n2b = {n: H.degree(n, s=1) for n in H.nodes}
	e2n_new = after_failures(e2n, p)
	H_new = hnx.Hypergraph(e2n_new)
	n2d = {n: H_new.degree(n, s=1) for n in H_new.nodes}
	failed_es2need = {e: R - len(ns) for e, ns in e2n_new.items() if len(ns) < R}
	patching_agents2supply = {n: B - d for n, d in n2d.items() if d < B}
	# patching
	patching_cost = 0
	penalty_cost = 0
	if not failed_es2need:
		return patching_cost + penalty_cost
	for fe, need in failed_es2need.items():
		pa2dis = {pa: distance_shortest(H_new, pa, fe, e2n_new) for pa in patching_agents2supply.keys()}
		for pa, dis in dict(sorted(pa2dis.items(), key=lambda x: x[1])).items():
			if dis > diameter:
				penalty_cost += diameter*2
				break
			if patching_agents2supply[pa] == need:
				patching_cost += dis
				del patching_agents2supply[pa]
				break
			if patching_agents2supply[pa] > need:
				patching_cost += dis
				patching_agents2supply[pa] = patching_agents2supply[pa] - need
				break
			if patching_agents2supply[pa] < need:
				patching_cost += dis
				need = need - patching_agents2supply[pa]
				del patching_agents2supply[pa]
				
	
	return patching_cost + penalty_cost
	
	
def retrive_NTBR(p):
	path = Path(p)
	data = []
	for line in path.read_text().splitlines():
		data.append([float(x) for x in line.split()])
	n_authors = len(data)
	n_papers = len(data[0])    
	count_b = 0
	for i in range(n_authors):
		count_b += sum([data[i][j] for j in range(n_papers) if data[i][j] != 0])
		
	count_r = 0
	for j in range(n_papers):
		count_r += sum([data[i][j] for i in range(n_authors) if data[i][j] != 0])
	return n_authors, n_papers, math.ceil(count_b / n_authors), math.ceil(count_r / n_papers)
	
	
def saving_compare_results_aps(year, type_ana, p_interval, runstep, results, results_aps):
	save_dir = f"results_compare_aps/{type_ana}"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"results_aps": results_aps,
		"p_interval": p_interval,
		"runstep": runstep,
		"year": year,
		"type_ana": type_ana
	}
	filename = f"{year}_{type_ana}_{num}_{runstep}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
	
def compare_aps(compare_types, p_interval, runstep):
	aps_dir = 'APS_data'
	ps = generate_interval(p_interval)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	for year in range(1994, 2021):
		file_name = 'LCC_PRE_' + f'{year}' + '_to_' + f'{year+1}' + ".txt"
		filepath = os.path.join(aps_dir, file_name)
		N, T, B, R = retrive_NTBR(filepath)
		H_aps = retrive_H_aps(filepath)
		R_aps = mean_edges_size(H_aps)
		B_aps = math.ceil(mean_degree(H_aps))
		if N*B_aps < T*R_aps:
			print(f'Not feasible for {year}-{year+1}')
			continue
		H = gene_fcr(N, T, B_aps, R_aps)
		if 'ana_11' in compare_types:
			results = Parallel(n_jobs=8)(delayed(unfeasi_num)(H, R_aps, p) for p, i in tasks)
			results_aps = Parallel(n_jobs=8)(delayed(unfeasi_num_aps)(H_aps, R_aps, p) for p, i in tasks)
			saving_compare_results_aps(year, 'ana_11', p_interval, runstep, results, results_aps)
		if 'ana_12' in compare_types:
			results = Parallel(n_jobs=8)(delayed(cc_num)(H, p) for p, i in tasks)
			results_aps = Parallel(n_jobs=8)(delayed(cc_num)(H_aps, p) for p, i in tasks)
			saving_compare_results_aps(year, 'ana_12', p_interval, runstep, results, results_aps)
		if 'ana_2' in compare_types:
			results = Parallel(n_jobs=8)(delayed(recovery_dis)(H, B_aps, R_aps, p) for p, i in tasks)
			results_aps = Parallel(n_jobs=8)(delayed(recovery_dis)(H_aps, B_aps, R_aps, p) for p, i in tasks)
			saving_compare_results_aps(year, 'ana_2', p_interval, runstep, results, results_aps)
		
def generate_uer(N, T, B, R):
	while True:
		hyperedgeList = [random.sample(range(N), R) for i in range(T)]
		H_uer = hnx.Hypergraph(hyperedgeList)
		if component_num(H_uer) == 1:
			break
	return H_uer	
		
		
def compare_er(N, T, B, R, compare_types, p_interval, runstep):
	ps = generate_interval(p_interval)
	H = gene_fcr(N, T, B, R)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	H_er_list_ = [generate_uer(N, T, B, R) for _ in range(runstep)]
	H_er_list = random.sample(H_er_list_, runstep)
	tasks_er = [(p, H_er) for p in ps for H_er in H_er_list]
	if 'ana_11' in compare_types:
		results = Parallel(n_jobs=8)(delayed(unfeasi_num)(H, R, p) for p, i in tasks)
		results_er = Parallel(n_jobs=8)(delayed(unfeasi_num)(H_er, R, p) for p, H_er in tasks_er)
		saving_compare_results_er('ana_11', p_interval, runstep, results, results_er)
	if 'ana_12' in compare_types:
		results = Parallel(n_jobs=8)(delayed(cc_num)(H, p) for p, i in tasks)
		results_er = Parallel(n_jobs=8)(delayed(cc_num)(H_er, p) for p, H_er in tasks_er)
		saving_compare_results_er('ana_12', p_interval, runstep, results, results_er)
	if 'ana_2' in compare_types:
		results = Parallel(n_jobs=8)(delayed(recovery_dis)(H, B, R, p) for p, i in tasks)
		results_er = Parallel(n_jobs=8)(delayed(recovery_dis)(H_er, B, R, p) for p, H_er in tasks_er)
		saving_compare_results_er('ana_2', p_interval, runstep, results, results_er)

def saving_compare_results_er(type_ana, p_interval, runstep, results, results_er):
	save_dir = f"results_compare_er/{type_ana}"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"results_er": results_er,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{num}_{runstep}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def select_nodes_by_degree(h, H):
	degrees = [H.degree(n, s=1) for n in H.nodes]
	total_deg = sum(degrees)
	probs = [d / total_deg for d in degrees]
	selected_nodes = list(np.random.choice(H.nodes, size=h, replace=False, p=probs))
	return selected_nodes


def new_edges(node, m, current_edges, k):
	H_current = hnx.Hypergraph(current_edges)
	return [select_nodes_by_degree(k-1, H_current) + [node] for _ in range(m)]
	
# uniform BA
def generate_uba(N, T, B, R):
	
	m = math.ceil(B / R)
	# start with a fully connected core - simplicial complex - size M 
	M = max([m-1, R])
	# initial hyperedges
	hyperedges = [list(pair) for pair in combinations(range(M), R)]
	while True:
	# 连续添加新节点直至节点数达到 N
		for t in range(M, N):
			added_edges = new_edges(t, 1, hyperedges, R)
			hyperedges.extend(added_edges)
			if len(hyperedges) > T:
				break
		if len(hyperedges) < T:
			hyperedges.extend([random.sample(range(N), R) for _ in range(T-len(hyperedges))])
		H_ba = hnx.Hypergraph(hyperedges)
		if component_num(H_ba) == 1:
			break
	return H_ba
	
def saving_compare_results_ba(type_ana, p_interval, runstep, results, results_ba):
	save_dir = f"results_compare_ba/{type_ana}"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"results_ba": results_ba,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{num}_{runstep}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def saving_compare_results_erba(type_ana, p_interval, runstep, results, results_er, results_ba):
	save_dir = f"results_compare_erba/{type_ana}"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"results_er": results_er,
		"results_ba": results_ba,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{num}_{runstep}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def compare_ba(N, T, B, R, compare_types, p_interval, runstep):
	ps = generate_interval(p_interval)
	H_ba = generate_uba(N, T, B, R)
	if 'ana_11' in compare_types:
		results = []
		results_ba = Parallel(n_jobs=8)(delayed(unfeasi_num)(H_ba, R, p) for p, i in tasks)
		saving_compare_results_ba('ana_11', p_interval, runstep, results, results_ba)
	if 'ana_12' in compare_types:
		results = []		
		results_ba = Parallel(n_jobs=8)(delayed(cc_num)(H_ba, p) for p, i in tasks)
		saving_compare_results_ba('ana_12', p_interval, runstep, results, results_ba)
	if 'ana_2' in compare_types:
		results = []
		results_ba = Parallel(n_jobs=8)(delayed(recovery_dis)(H_ba, B, R, p) for p, i in tasks)
		saving_compare_results_ba('ana_2', p_interval, runstep, results, results_ba)
	
	
def compare_erba(N, T, B, R, compare_types, p_interval, runstep):
	ps = generate_interval(p_interval)
	H = gene_fcr(N, T, B, R)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	H_ba = generate_uba(N, T, B, R)
	H_er_list = [generate_uer(N, T, B, R) for _ in range(runstep)]
	tasks_er = [(p, H_er) for p in ps for H_er in H_er_list]
	if 'ana_11' in compare_types:
		results = Parallel(n_jobs=4)(delayed(unfeasi_num)(H, R, p) for p, i in tasks)
		results_er = Parallel(n_jobs=4)(delayed(unfeasi_num)(H_er, R, p) for p, H_er in tasks_er)
		results_ba = Parallel(n_jobs=4)(delayed(unfeasi_num)(H_ba, R, p) for p, i in tasks)
		saving_compare_results_erba('ana_11', p_interval, runstep, results, results_er, results_ba)
	if 'ana_12' in compare_types:
		results = Parallel(n_jobs=8)(delayed(cc_num)(H, p) for p, i in tasks)
		results_er = Parallel(n_jobs=8)(delayed(cc_num)(H_er, p) for p, H_er in tasks_er)
		results_ba = Parallel(n_jobs=8)(delayed(cc_num)(H_ba, p) for p, i in tasks)
		saving_compare_results_erba('ana_12', p_interval, runstep, results, results_er, results_ba)
	if 'ana_2' in compare_types:
		results = [Parallel(n_jobs=8)(delayed(recovery_dis)(H, B, R, p) for p, i in tasks)]
		results_er = Parallel(n_jobs=8)(delayed(recovery_dis)(H_er, B, R, p) for p, H_er in tasks_er)
		results_ba = Parallel(n_jobs=8)(delayed(recovery_dis)(H_ba, B, R, p) for p, i in tasks)
		saving_compare_results_erba('ana_2', p_interval, runstep, results, results_er, results_ba)
		
		
def compare_sa(N, T, B, R, compare_types, p_interval, runstep):
	ps = generate_interval(p_interval)
	H = gene_fcr(N, T, B, R)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	H_sa_list = [sa.simulated_annealing(N, T, B, R) for _ in range(runstep)]
	tasks_sa = [(p, H_sa) for p in ps for H_sa in H_sa_list]
	if 'ana_11' in compare_types:
		results = Parallel(n_jobs=8)(delayed(unfeasi_num)(H, R, p) for p, i in tasks)
		results_sa = Parallel(n_jobs=8)(delayed(unfeasi_num)(H_sa, R, p) for p, H_sa in tasks_sa)
		saving_compare_results_sa('ana_11', p_interval, runstep, results, results_sa)
	if 'ana_12' in compare_types:
		results = Parallel(n_jobs=8)(delayed(cc_num)(H, p) for p, i in tasks)
		results_sa = Parallel(n_jobs=8)(delayed(cc_num)(H_sa, p) for p, H_sa in tasks_sa)
		saving_compare_results_sa('ana_12', p_interval, runstep, results, results_sa)
	if 'ana_2' in compare_types:
		results = Parallel(n_jobs=8)(delayed(recovery_dis)(H, B, R, p) for p, i in tasks)
		results_sa = Parallel(n_jobs=8)(delayed(recovery_dis)(H_sa, B, R, p) for p, H_sa in tasks_sa)
		saving_compare_results_sa('ana_2', p_interval, runstep, results, results_sa)
		

def saving_compare_results_sa(type_ana, p_interval, runstep, results, results_sa):
	save_dir = f"results_compare_sa/{type_ana}"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"results_sa": results_sa,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{num}_{runstep}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")

def saving_single_sa(type_sa, type_ana, p_interval, runstep, results_sa):
	save_dir = f"results_single_{type_sa}/{type_ana}"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results_sa": results_sa,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{num}_{runstep}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")

def mu2_sa(N, T, B, R, compare_types, p_interval, runstep):
	ps = generate_interval(p_interval)
	H_sa = mu2.simulated_annealing(N, T, B, R)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	if 'ana_11' in compare_types:
		results_sa = Parallel(n_jobs=8)(delayed(unfeasi_num)(H_sa, R, p) for p, i in tasks)
		saving_single_sa('mu2', 'ana_11', p_interval, runstep, results_sa)
	if 'ana_12' in compare_types:
		results_sa = Parallel(n_jobs=8)(delayed(cc_num)(H_sa, p) for p, i in tasks)
		saving_single_sa('mu2', 'ana_12', p_interval, runstep, results_sa)
	if 'ana_2' in compare_types:
		results_sa = Parallel(n_jobs=8)(delayed(recovery_dis)(H_sa, B, R, p) for p, i in tasks)
		saving_single_sa('mu2', 'ana_2', p_interval, runstep, results_sa)
	
def csa_redo11(N, T, B, R, compare_types, p_interval, runstep):
	ps = generate_interval(p_interval)
	H_sa_list = [sa.simulated_annealing(N, T, B, R) for _ in range(runstep)]
	H_sa_tasks = [(p, H) for p in ps for H in H_sa_list]
	if 'ana_11' in compare_types:
		results_sa = Parallel(n_jobs=8)(delayed(unfeasi_num_aps)(H_sa, R, p) for p, H_sa in H_sa_tasks)
		saving_single_sa('csa', 'ana_11', p_interval, runstep, results_sa)
	
	
def saving_single_gr(type_ana, p_interval, runstep, results):
	save_dir = f"results_single_greedy/{type_ana}"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results_gr": results,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{num}_{runstep}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def run_greedy_repair(N, T, B, R, compare_types, p_interval, runstep):
	ps = generate_interval(p_interval)
	H_gr_list = [gr.greedy(N, T, B, R) for _ in range(runstep)]
	H_gr_tasks = [(p, H) for p in ps for H in H_gr_list]
	if 'ana_11' in compare_types:
		results = Parallel(n_jobs=8)(delayed(unfeasi_num)(H, R, p) for p, H in H_gr_tasks)
		saving_single_gr('ana_11', p_interval, runstep, results)
	if 'ana_12' in compare_types:
		results = Parallel(n_jobs=8)(delayed(cc_num)(H, p) for p, H in H_gr_tasks)
		saving_single_gr('ana_12', p_interval, runstep, results)
	if 'ana_2' in compare_types:
		results = Parallel(n_jobs=8)(delayed(recovery_dis)(H, B, R, p) for p, H in H_gr_tasks)
		saving_single_gr('ana_2', p_interval, runstep, results)
	
	
	
	
def get_runtime(N, T, B, R):
	save_dir = "results_runtime"
	os.makedirs(save_dir, exist_ok=True)

	results = {}
	
	time0 = time.time()
	H = gene_fcr(N, T, B, R)
	time1 = time.time()
	results['star'] = time1 - time0
	
	time0 = time.time()
	H = generate_uer(N, T, B, R)
	time1 = time.time()
	results['er'] = time1 - time0
	
	time0 = time.time()
	H = generate_uba(N, T, B, R)
	time1 = time.time()
	results['ba'] = time1 - time0
	
	time0 = time.time()
	H = sa.simulated_annealing(N, T, B, R)
	time1 = time.time()
	results['sa'] = time1 - time0
	
	time0 = time.time()
	H = gr.greedy(N, T, B, R)
	time1 = time.time()
	results['gr'] = time1 - time0
	
	filename = f"{N}_{T}_{B}_{R}" + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(results, f, indent=2)
	print(f"结果保存到: {filepath}")
	
	
def get_solutions_diffb(paras):
	solutions, solutions_er, solutions_ba = [], [], []
	solutions_gr, solutions_sa, solutions_mu2 = [], [], []

	for para in paras:
		H = gene_fcr(**para)
		H_mu2 = mu2.simulated_annealing(100, 100, para['B'], 3)
		H_er = [generate_uer(**para) for _ in range(10)]
		H_ba = [generate_uba(**para) for _ in range(10)]
		H_sa = [sa.simulated_annealing(100, 100, para['B'], 3) for _ in range(10)]
		H_gr = [gr.greedy(100, 100, para['B'], 3) for _ in range(10)]

		solutions.append(H)
		solutions_er.append(H_er)
		solutions_ba.append(H_ba)
		solutions_gr.append(H_gr)
		solutions_sa.append(H_sa)
		solutions_mu2.append(H_mu2)

	return solutions, solutions_er, solutions_ba, solutions_gr, solutions_sa, solutions_mu2
	
def retrive_data(sols):
	diameters = [H.edge_diameter() for H in sols]
	Xs = [H2X(H) for H in sols]
	acs = [algebraic_connectivity(X) for X in Xs]

	return diameters, acs

def retrive_data_multi(sols):
	dias, acs = [], []
	for sols_list in sols:
		dia = [H.edge_diameter() for H in sols_list]
		Xs = [H2X(H) for H in sols_list]
		ac = [algebraic_connectivity(X) for X in Xs]
		mean_dia = sum(dia) / len(dia)
		mean_ac = sum(ac) / len(ac)
		dias.append(mean_dia)
		acs.append(mean_ac)
	return dias, acs

def run_save(paras, info):
	data = {
		"H_": {},
		"H_er": {},
		"H_ba": {},
		"H_sa": {},
		"H_gr": {},
		"H_mu2": {}
	}
	save_dir = f"results_scatter/"
	os.makedirs(save_dir, exist_ok=True)
	solutions, solutions_er, solutions_ba, solutions_gr, solutions_sa, solutions_mu2 = get_solutions_diffb(paras)

	dias, acs = retrive_data(solutions)
	data['H_']['dia'], data['H_']['ac'] = dias, acs
	
	dias, acs = retrive_data_multi(solutions_er)
	data['H_er']['dia'], data['H_er']['ac'] = dias, acs    
	
	dias, acs = retrive_data_multi(solutions_ba)
	data['H_ba']['dia'], data['H_ba']['ac'] = dias, acs
	
	dias, acs = retrive_data_multi(solutions_gr)
	data['H_gr']['dia'], data['H_gr']['ac'] = dias, acs

	dias, acs = retrive_data_multi(solutions_sa)
	data['H_sa']['dia'], data['H_sa']['ac'] = dias, acs
	
	dias, acs = retrive_data(solutions_mu2)
	data['H_mu2']['dia'], data['H_mu2']['ac'] = dias, acs
	
	filename = f"{info}" + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + '.json'
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def retrive_Hs_determined(N, T, B, R):
	H_star = gene_fcr(N, T, B, R)
	H_mu2 = mu2.simulated_annealing(N, T, B, R)
#	H_er = generate_uer(N, T, B, R)
#	H_ba = generate_uba(N, T, B, R)
#	H_sa = sa.simulated_annealing(N, T, B, R)
#	H_gr = gr.greedy(N, T, B, R)
	return [H_star, H_mu2]
	
	
	
	
		
	
	
def save_H_star(N, T, B, R):
	H = gene_fcr(N, T, B, R)
	e2n = H.incidence_dict
	save_dir = f"results_H/star"
	os.makedirs(save_dir, exist_ok=True)
	filename = f"{N}_{T}_{B}_{R}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(e2n, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def save_H_drop1(N, T, B, R):
	H = gene_fcr_drop1(N, T, B, R)
	e2n = H.incidence_dict
	save_dir = f"results_H/drop1"
	os.makedirs(save_dir, exist_ok=True)
	filename = f"{N}_{T}_{B}_{R}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(e2n, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def save_H_drop2(N, T, B, R):
	H = gene_fcr_drop2(N, T, B, R)
	e2n = H.incidence_dict
	save_dir = f"results_H/drop2"
	os.makedirs(save_dir, exist_ok=True)
	filename = f"{N}_{T}_{B}_{R}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(e2n, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def save_H_drop12(N, T, B, R):
	H = gene_fcr_drop12(N, T, B, R)
	e2n = H.incidence_dict
	save_dir = f"results_H/drop12"
	os.makedirs(save_dir, exist_ok=True)
	filename = f"{N}_{T}_{B}_{R}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(e2n, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def save_H_er(N, T, B, R):
	H = generate_uer(N, T, B, R)
	e2n = H.incidence_dict
	save_dir = f"results_H/er"
	os.makedirs(save_dir, exist_ok=True)
	filename = f"{N}_{T}_{B}_{R}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(e2n, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def save_H_ba(N, T, B, R):
	H = generate_uba(N, T, B, R)
	e2n = H.incidence_dict
	save_dir = f"results_H/ba"
	os.makedirs(save_dir, exist_ok=True)
	filename = f"{N}_{T}_{B}_{R}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(e2n, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def save_H_mu2(N, T, B, R):
	H = mu2.simulated_annealing(N, T, B, R)
	e2n = H.incidence_dict
	save_dir = f"results_H/mu2"
	os.makedirs(save_dir, exist_ok=True)
	filename = f"{N}_{T}_{B}_{R}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(e2n, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def save_H_sa(N, T, B, R):
	H = mu2.simulated_annealing(N, T, B, R)
	e2n = H.incidence_dict
	save_dir = f"results_H/sa"
	os.makedirs(save_dir, exist_ok=True)
	filename = f"{N}_{T}_{B}_{R}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(e2n, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def save_H_gr(N, T, B, R):
	H = gr.greedy(N, T, B, R)
	e2n = H.incidence_dict
	save_dir = f"results_H/gr"
	os.makedirs(save_dir, exist_ok=True)
	filename = f"{N}_{T}_{B}_{R}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(e2n, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def retrive_from_file(paths):
	Hs = []
	for p in paths:
		with open(p, 'r', encoding='utf-8') as f:
			data = json.load(f)
		H = hnx.Hypergraph(data)
		Hs.append(H)
	return Hs
	
def saving_k(k, results, ana, p_interval, runstep):
	save_dir = f"results_compare_sa/{ana}"
	os.makedirs(save_dir, exist_ok=True)
#	H_names = ['star', 'mu2', 'sa', 'gr']
	H_names = ['star', 'star', 'star', 'star']
	H_name = H_names[k]
	filename = f"{H_name}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	data = {
		"results": results,
		"ana": ana,
		"p_interval": list(p_interval),
		"runstep": runstep
	}
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def compare_all4(paths, compare_types, p_interval, runstep, R, B):
	ps = generate_interval(p_interval)
	Hs = retrive_from_file(paths)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	if 'ana_11' in compare_types:
		for k, H in enumerate(Hs):
			results = Parallel(n_jobs=8)(delayed(unfeasi_num)(H, R, p) for p, i in tasks)
			saving_k(k, results, 'ana_11', p_interval, runstep)
	
	if 'ana_12' in compare_types:
		for k, H in enumerate(Hs):
			results = Parallel(n_jobs=8)(delayed(cc_num)(H, p) for p, i in tasks)
			saving_k(k, results, 'ana_12', p_interval, runstep)
			
	if 'ana_2' in compare_types:
		for k, H in enumerate(Hs):
			results = Parallel(n_jobs=8)(delayed(recovery_dis)(H, B, R, p) for p, i in tasks)
			saving_k(k, results, 'ana_2', p_interval, runstep)
	
	
def saving_H_aps():
	aps_dir = 'APS_data/'
	years = [1994, 1996, 1999, 2001, 2002, 2003, 2004, 2007, 2009, 2013, 2014, 2016, 2017, 2018, 2019, 2020]
	for year in years:
		file_name = 'LCC_PRE_' + f'{year}' + '_to_' + f'{year+1}' + ".txt"
		filepath = os.path.join(aps_dir, file_name)
		N, T, B, R = retrive_NTBR(filepath)
		H_aps = retrive_H_aps(filepath)
		R_aps = mean_edges_size(H_aps)
		B_aps = math.ceil(mean_degree(H_aps))
		H = gene_fcr(N, T, B_aps, R_aps)
		e2n_aps = H_aps.incidence_dict
		e2n = H.incidence_dict
		data = {'aps': e2n_aps, 'H': e2n}
		save_dir = f"results_H/aps"
		os.makedirs(save_dir, exist_ok=True)
		filename = f"{year}" + ".json"
		filepath = os.path.join(save_dir, filename)
		with open(filepath, "w") as f:
			json.dump(data, f, indent=2)
		print(f"结果保存到: {filepath}")
		
def compare_aps_cloud(compare_types, p_interval, runstep):
	years = [2001, 2002, 2003, 2004, 2007, 2009, 2013, 2014, 2016, 2017, 2018, 2019, 2020]
	ps = generate_interval(p_interval)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	for year in years:
		path = f"{year}" + ".json"
		with open(path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		H = hnx.Hypergraph(data['H'])
		H_aps = hnx.Hypergraph(data['aps'])
		R_aps = mean_edges_size(H_aps)
		B_aps = math.ceil(mean_degree(H_aps))
		results = Parallel(n_jobs=-1)(delayed(recovery_dis)(H, B_aps, R_aps, p) for p, i in tasks)
		results_aps = Parallel(n_jobs=-1)(delayed(recovery_dis)(H_aps, B_aps, R_aps, p) for p, i in tasks)
		saving_compare_results_aps(year, 'ana_2', p_interval, runstep, results, results_aps)
		
	
def saving_drop1(type_ana, p_interval, runstep, results):
	save_dir = f"results_ablation/drop1"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{type_ana}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	

	
def saving_drop2(type_ana, p_interval, runstep, results):
	save_dir = f"results_ablation/drop2"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{type_ana}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def saving_drop12(type_ana, p_interval, runstep, results):
	save_dir = f"results_ablation/drop12"
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{type_ana}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def retrive_H_one(p):
	with open(p, 'r', encoding='utf-8') as f:
			data = json.load(f)
	return hnx.Hypergraph(data)
	
	
	
def run_ablation_drop1(N, T, B, R, compare_types, p_interval, runstep):
	ps = generate_interval(p_interval)
	H_list = Parallel(n_jobs=8)(delayed(gene_fcr_drop1)(N, T, B, R) for _ in range(runstep))
	H_tasks = [(p, H) for p in ps for H in H_list]	
	if 'ana_11' in compare_types:
		results = Parallel(n_jobs=8)(delayed(unfeasi_num)(H, R, p) for p, H in H_tasks)
		saving_drop1('ana_11', p_interval, runstep, results)
	if 'ana_12' in compare_types:
		results = Parallel(n_jobs=8)(delayed(cc_num)(H, p) for p, H in H_tasks)
		saving_drop1('ana_12', p_interval, runstep, results)
	if 'ana_2' in compare_types:
		results = Parallel(n_jobs=8)(delayed(recovery_dis)(H, B, R, p) for p, H in H_tasks)
		saving_drop1('ana_2', p_interval, runstep, results)
		
def saving_ablation(k, type_ana, p_interval, runstep, results):
	save_dir = f"results_ablation/{type_ana}"
	drop_name = ['drop1', 'drop2', 'drop12']
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{drop_name[k]}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
		
def run_ablation_new(paths, B, R, compare_types, p_interval, runstep):
	ps = generate_interval(p_interval)
	# 1 2 12
	Hs = retrive_from_file(paths)
	H_tasks = [(p, i) for p in ps for i in range(runstep)]	
	if 'ana_11' in compare_types:
		for k, H in enumerate(Hs):
			results = Parallel(n_jobs=8)(delayed(unfeasi_num)(H, R, p) for p, i in H_tasks)
			saving_ablation(k, 'ana_11', p_interval, runstep, results)
		
	if 'ana_12' in compare_types:
		for k, H in enumerate(Hs):
			results = Parallel(n_jobs=8)(delayed(cc_num)(H, p) for p, i in H_tasks)
			saving_ablation(k, 'ana_12', p_interval, runstep, results)
			
	if 'ana_2' in compare_types:
		for k, H in enumerate(Hs):
			results = Parallel(n_jobs=8)(delayed(recovery_dis)(H, B, R, p) for p, i in H_tasks)
			saving_ablation(k, 'ana_2', p_interval, runstep, results)

	
def run_ablation_drop2(N, T, B, R, compare_types, p_interval, runstep):
	ps = generate_interval(p_interval)
	H_list = Parallel(n_jobs=8)(delayed(gene_fcr_drop2)(N, T, B, R) for _ in range(runstep))
	H_tasks = [(p, H) for p in ps for H in H_list]	
	if 'ana_11' in compare_types:
		results = Parallel(n_jobs=8)(delayed(unfeasi_num)(H, R, p) for p, H in H_tasks)
		saving_drop2('ana_11', p_interval, runstep, results)
	if 'ana_12' in compare_types:
		results = Parallel(n_jobs=8)(delayed(cc_num)(H, p) for p, H in H_tasks)
		saving_drop2('ana_12', p_interval, runstep, results)
	if 'ana_2' in compare_types:
		results = Parallel(n_jobs=8)(delayed(recovery_dis)(H, B, R, p) for p, H in H_tasks)
		saving_drop2('ana_2', p_interval, runstep, results)
		
def run_ablation_drop12(N, T, B, R, compare_types, p_interval, runstep):
	ps = generate_interval(p_interval)
	H_list = Parallel(n_jobs=8)(delayed(gene_fcr_drop12)(N, T, B, R) for _ in range(runstep))
	H_tasks = [(p, H) for p in ps for H in H_list]	
	if 'ana_11' in compare_types:
		results = Parallel(n_jobs=8)(delayed(unfeasi_num)(H, R, p) for p, H in H_tasks)
		saving_drop12('ana_11', p_interval, runstep, results)
	if 'ana_12' in compare_types:
		results = Parallel(n_jobs=8)(delayed(cc_num)(H, p) for p, H in H_tasks)
		saving_drop12('ana_12', p_interval, runstep, results)
	if 'ana_2' in compare_types:
		results = Parallel(n_jobs=8)(delayed(recovery_dis)(H, B, R, p) for p, H in H_tasks)
		saving_drop12('ana_2', p_interval, runstep, results)
	
def saving_sensir(k, type_ana, p_interval, runstep, results):
	save_dir = f"results_sensir/{type_ana}"
	Rs = [2, 3, 4, 5]
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{Rs[k]}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def sensi_r(paths, compare_types, p_interval, runstep, Rs, Bs):
	ps = generate_interval(p_interval)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	Hs = retrive_from_file(paths)
	if 'ana_11' in compare_types:
		for k in range(4):
			results = Parallel(n_jobs=8)(delayed(unfeasi_num)(Hs[k], Rs[k], p) for p, i in tasks)
			saving_sensir(k, 'ana_11', p_interval, runstep, results)
	if 'ana_12' in compare_types:
		for k in range(4):
			results = Parallel(n_jobs=8)(delayed(cc_num)(Hs[k], p) for p, i in tasks)
			saving_sensir(k, 'ana_12', p_interval, runstep, results)
	if 'ana_2' in compare_types:
		for k in range(4):
			results = Parallel(n_jobs=8)(delayed(recovery_dis)(Hs[k], Bs[k], Rs[k], p) for p, i in tasks)
			saving_sensir(k, 'ana_2', p_interval, runstep, results)
	
def saving_sensint(k, type_ana, p_interval, runstep, results):
	save_dir = f"results_sensint/{type_ana}"
	Rs = [0.5, 1.0, 2.0]
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{Rs[k]}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def sensi_nt(paths, compare_types, p_interval, runstep, Rs, Bs):
	ps = generate_interval(p_interval)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	Hs = retrive_from_file(paths)
	if 'ana_11' in compare_types:
		for k in range(3):
			results = Parallel(n_jobs=8)(delayed(unfeasi_num)(Hs[k], Rs[k], p) for p, i in tasks)
			saving_sensint(k, 'ana_11', p_interval, runstep, results)
	if 'ana_12' in compare_types:
		for k in range(3):
			results = Parallel(n_jobs=8)(delayed(cc_num)(Hs[k], p) for p, i in tasks)
			saving_sensint(k, 'ana_12', p_interval, runstep, results)
	if 'ana_2' in compare_types:
		for k in range(3):
			results = Parallel(n_jobs=8)(delayed(recovery_dis)(Hs[k], Bs[k], Rs[k], p) for p, i in tasks)
			saving_sensint(k, 'ana_2', p_interval, runstep, results)
			
def saving_sensint_diff(k, type_ana, p_interval, runstep, results):
	save_dir = f"results_sensint_diff/{type_ana}"
	diffs = ['star', 'sa', 'mu2', 'gr']
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{diffs[k]}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def sensi_nt_diff(paths, compare_types, p_interval, runstep, Rs, Bs):
	ps = generate_interval(p_interval)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	Hs = retrive_from_file(paths)
	if 'ana_11' in compare_types:
		for k in range(3):
			results = Parallel(n_jobs=8)(delayed(unfeasi_num)(Hs[k], Rs[k], p) for p, i in tasks)
			saving_sensint_diff(k, 'ana_11', p_interval, runstep, results)
	if 'ana_12' in compare_types:
		for k in range(3):
			results = Parallel(n_jobs=8)(delayed(cc_num)(Hs[k], p) for p, i in tasks)
			saving_sensint_diff(k, 'ana_12', p_interval, runstep, results)
	if 'ana_2' in compare_types:
		for k in range(3):
			results = Parallel(n_jobs=8)(delayed(recovery_dis)(Hs[k], Bs[k], Rs[k], p) for p, i in tasks)
			saving_sensint_diff(k, 'ana_2', p_interval, runstep, results)
			
			
def saving_sensir_all(k, type_ana, p_interval, runstep, results):
	save_dir = f"results_sensir/{type_ana}"
	syurui = ['sa', 'gr', 'mu2', 'ba', 'er']
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{syurui[k]}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
			
def sensi_r_all(paths, compare_types, p_interval, runstep, R, B):
	ps = generate_interval(p_interval)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	Hs = retrive_from_file(paths)
	if 'ana_11' in compare_types:
		for k in range(5):
			results = Parallel(n_jobs=8)(delayed(unfeasi_num)(Hs[k], R, p) for p, i in tasks)
			saving_sensir_all(k, 'ana_11', p_interval, runstep, results)
	if 'ana_12' in compare_types:
		for k in range(5):
			results = Parallel(n_jobs=8)(delayed(cc_num)(Hs[k], p) for p, i in tasks)
			saving_sensir_all(k, 'ana_12', p_interval, runstep, results)
	if 'ana_2' in compare_types:
		for k in range(5):
			results = Parallel(n_jobs=8)(delayed(recovery_dis)(Hs[k], B, R, p) for p, i in tasks)
			saving_sensir_all(k, 'ana_2', p_interval, runstep, results)
			
def saving_sensint_diff_all(k, type_ana, p_interval, runstep, results):
	save_dir = f"results_sensint_diff/{type_ana}"
	diffs = ['star', 'sa', 'gr', 'mu2', 'ba', 'er']
	os.makedirs(save_dir, exist_ok=True)
	stop, num = p_interval
	data = {
		"results": results,
		"p_interval": p_interval,
		"runstep": runstep,
		"type_ana": type_ana
	}
	filename = f"{diffs[k]}".replace(".", "_") + 't' +  datetime.now().isoformat().split(":")[-1].replace(".", "_") + ".json"
	filepath = os.path.join(save_dir, filename)
	with open(filepath, "w") as f:
		json.dump(data, f, indent=2)
	print(f"结果保存到: {filepath}")
	
def sensi_nt_diff_all(paths, compare_types, p_interval, runstep, R, B):
	ps = generate_interval(p_interval)
	tasks = [(p, i) for p in ps for i in range(runstep)]
	Hs = retrive_from_file(paths)
	if 'ana_11' in compare_types:
		for k in range(6):
			results = Parallel(n_jobs=56)(delayed(unfeasi_num)(Hs[k], R, p) for p, i in tasks)
			saving_sensint_diff_all(k, 'ana_11', p_interval, runstep, results)
	if 'ana_12' in compare_types:
		for k in range(6):
			results = Parallel(n_jobs=56)(delayed(cc_num)(Hs[k], p) for p, i in tasks)
			saving_sensint_diff_all(k, 'ana_12', p_interval, runstep, results)
	if 'ana_2' in compare_types:
		for k in range(6):
			results = Parallel(n_jobs=56)(delayed(recovery_dis)(Hs[k], B, R, p) for p, i in tasks)
			saving_sensint_diff_all(k, 'ana_2', p_interval, runstep, results)
