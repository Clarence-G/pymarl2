import oapackage


def separate_pareto_frontier(p, func=lambda x: x):
    pareto = oapackage.ParetoDoubleLong()
    for i in range(len(p)):
        w = oapackage.doubleVector(func(p[i]))
        pareto.addvalue(w, i)
    lst = pareto.allindices()
    opt = []
    for i in range(len(lst)):
        opt.append(p[lst[i]])
    non_opt = []
    for i in range(len(p)):
        if i not in lst:
            non_opt.append(p[i])
    return opt, non_opt


def cal_all_pareto_frontier(p, func=lambda x: x):
    fronts = []
    while len(p) > 0:
        opt, non_opt = separate_pareto_frontier(p, func)
        fronts.append(opt)
        p = non_opt

    return fronts
