"""Development-only descriptive EDA; TEST receives structural counts only."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core as c

plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False,'font.size':9,'savefig.bbox':'tight'})
COLORS={'representative':'#235c88','stress':'#d9832e'}


def quantiles(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    return dict(zip(['min','p10','median','p90','max'],map(float,np.quantile(a,[0,.1,.5,.9,1])))) if len(a) else None

def cramers_v(a,b):
    t=pd.crosstab(a,b).to_numpy(dtype=float);n=t.sum()
    if min(t.shape)<=1:return 0.
    expected=t.sum(1)[:,None]*t.sum(0)[None,:]/n
    return float(np.sqrt(np.sum((t-expected)**2/np.maximum(expected,1e-30))/(n*min(t.shape[0]-1,t.shape[1]-1))))

def md_table(frame):return frame.to_markdown(index=False,floatfmt='.4g')

def cluster_interval(frame,column,seed=20261025):
    means=frame.groupby('family_id')[column].mean().to_numpy();g=np.random.default_rng(seed)
    bs=np.array([g.choice(means,len(means),replace=True).mean() for _ in range(2000)])
    return {'mean':float(means.mean()),'low95':float(np.quantile(bs,.025)),'high95':float(np.quantile(bs,.975)),
            'families':len(means),'unit':'family, equal-weight means','interpretation':'descriptive bootstrap interval'}

def report(product,output):
    output.mkdir(parents=True,exist_ok=True);figdir=output/'figures';figdir.mkdir(exist_ok=True)
    manifest=json.loads((product/'DATASET_MANIFEST.json').read_text())
    families=json.loads((product/'families.json').read_text());roots=json.loads((product/'roots.json').read_text())
    fammap={f['family_id']:f for f in families};frame=pd.DataFrame([{k:v for k,v in f.items() if k not in ('admission','geometry')} for f in families])
    root_rows=[];cov_rows=[];truth_rows=[];bootstrap_rows=[];bootcorr=[];count_summaries=[];feature_correlations=[]
    selected_boot={next(r['root_id'] for r in roots if r['family_id']==f['family_id']) for f in families if f['partition']!='TEST'}
    max_boot_psd_error=0.;max_cov_psd_error=0.
    gate_counts=Counter();screen=json.loads((product/'screening_development.json').read_text())
    for root_index,root in enumerate(roots):
        if root['partition']=='TEST':continue  # no TEST arrays or truth opened by this EDA program
        fam=fammap[root['family_id']];rid=root['root_id']
        for name,_,_ in root['circuit']['operations']:gate_counts[name]+=1
        raw=np.load(product/root['raw_path'],allow_pickle=False);derived=np.load(product/root['derived_path'],allow_pickle=False)
        cc=raw['calibration_counts'];table=raw['outcome_features']
        j=c.counts_to_jacobian(cc,table);x=c.canonical_batch(derived['delta'],j,derived['weights'])
        profiles=c.profile_batch(derived['delta'],j,derived['weights'])
        for field,array in [('jacobian_sha256',j),('feature_sha256',x),('profile_sha256',profiles)]:
            if str(derived[field])!=c.digest(np.ascontiguousarray(array).tobytes()):raise RuntimeError('Derived recipe hash mismatch')
        truth=json.loads((product/'development_truth'/root['partition']/(rid[7:]+'.json')).read_text())['truth']
        for entry in truth:
            o=next(o for o in root['observations'] if o['observation_id']==entry['observation_id'])
            truth_rows.append({**entry,'root_id':rid,'family_id':root['family_id'],'partition':root['partition'],
                'sampling_stratum':root['sampling_stratum'],'n_qubits':root['n_qubits'],'shots':o['shots'],
                'topology_class':fam['topology_class'],'reference_layers':fam['reference_layers'],'context':fam['injection_context_class']})
        target=np.load(product/'acquisition_audit'/root['partition']/(rid[7:]+'.npz'),allow_pickle=False)['first_candidate_probabilities']
        ts=np.einsum('...o,ok->...k',target,table)
        target_j=np.moveaxis((ts[:,0]-ts[:,1])/.24,0,-1).reshape(-1,3)
        for b,shots in enumerate(c.BUDGETS):
            norms=np.linalg.norm(j[b],axis=-2);u=j[b]/np.maximum(norms,1e-12)[...,None,:]
            singular=np.linalg.svd(u,compute_uv=False)
            condition=singular[...,0]/np.maximum(singular[...,-1],1e-12)
            # Variation stats average candidates/observations within each root; circuits, not draws, are sampling units.
            coord_sd=np.std(x[b,...,:3],axis=0,ddof=1);matched_sd=np.std(x[b,...,9:12],axis=0,ddof=1)
            row={'root_id':rid,'family_id':root['family_id'],'partition':root['partition'],
                 'sampling_stratum':root['sampling_stratum'],'budget':shots,
                 'n_qubits':root['n_qubits'],'median_column_norm':float(np.median(norms)),
                 'median_smin':float(np.median(singular[...,-1])),'median_condition':float(np.median(condition)),
                 'coordinate_sd_mean':float(coord_sd.mean()),'matched_sd_mean':float(matched_sd.mean()),
                 'matched_sd_max':float(matched_sd.max()),'feature_abs_max':float(abs(x[b]).max()),
                 'count_zero_fraction':float((cc[b]==0).mean())}
            root_rows.append(row)
            empirical=np.var(j[b,:,0],axis=0,ddof=1)
            _,estimated=c.UNC.finite_difference_jacobian_covariance(cc[b,0,0],np.repeat(table[None,:,:],3,axis=0),c.EPSILON)
            trace_est=sum(float(np.trace(v)) for v in estimated);trace_emp=float(empirical.sum())
            # True acquisition covariance for evaluation only: means and probabilities never reach the estimator.
            true_var=np.zeros_like(target_j)
            for axis in range(3):
                for basis in range(3):
                    var=np.zeros(table.shape[1])
                    for sign in range(2):
                        p=target[axis,sign,basis];mean=p@table
                        var+=(p@(table*table)-mean**2)/shots/(.24**2)
                    true_var[basis*table.shape[1]:(basis+1)*table.shape[1],axis]=var
            bias=np.mean(j[b,:,0],axis=0)-target_j
            covered=abs(bias)<=1.96*np.sqrt(np.maximum(true_var,0)/manifest['draws_per_budget'])+1e-12
            valid=true_var>1e-12
            cov_rows.append({'root_id':rid,'family_id':root['family_id'],'budget':shots,
                'sampling_stratum':root['sampling_stratum'],'empirical_trace':trace_emp,'observed_estimated_trace':trace_est,
                'trace_ratio_empirical_over_estimated':trace_emp/max(trace_est,1e-30),
                'mean_absolute_jacobian_bias':float(abs(bias).mean()),
                'approx95_mean_j_coverage':float(covered[valid].mean()) if valid.any() else 1.,
                'true_variance_trace':float(true_var.sum())})
            for a in estimated:max_cov_psd_error=min(max_cov_psd_error,float(np.linalg.eigvalsh(a).min()))
            count_summaries.append({'budget':shots,'root_id':rid,'count_min':int(cc[b].min()),'count_max':int(cc[b].max()),
                                    'count_mean':float(cc[b].mean()),'zero_fraction':float((cc[b]==0).mean())})
        if rid in selected_boot:
            boot=c.bootstrap_joint(cc[0,0,0],table,derived['delta'][0],derived['weights'][0],64,c.seed('eda-bootstrap',rid))
            covariance=boot['covariance'];max_boot_psd_error=min(max_boot_psd_error,float(np.linalg.eigvalsh(covariance).min()))
            arr=np.column_stack([x[0,:,0,0,:3],x[0,:,0,0,9:12]])
            empirical_cov=np.cov(arr,rowvar=False,ddof=1)
            std=np.sqrt(np.maximum(np.diag(covariance),1e-30));corr=covariance/(std[:,None]*std[None,:]);bootcorr.append(corr)
            feature_correlations.append(np.corrcoef(arr,rowvar=False))
            bootstrap_rows.append({'root_id':rid,'family_id':root['family_id'],'sampling_stratum':root['sampling_stratum'],
                'replicates':64,'conditional_on_fixed_diagnostic':True,'covariance':covariance.tolist(),
                'empirical_acquisition_covariance':empirical_cov.tolist(),
                'coordinate_trace_ratio':float(np.trace(empirical_cov[:3,:3])/max(np.trace(covariance[:3,:3]),1e-30)),
                'matched_trace_ratio':float(np.trace(empirical_cov[3:,3:])/max(np.trace(covariance[3:,3:]),1e-30)),
                'max_abs_coordinate_matched_correlation':float(abs(corr[:3,3:]).max()),
                'empirical_max_abs_coordinate_matched_correlation':float(abs(np.nan_to_num(np.corrcoef(arr,rowvar=False)[:3,3:])).max())})
        if (root_index+1)%100==0:print(f'EDA processed {root_index+1} root positions (TEST skipped)',flush=True)
    r=pd.DataFrame(root_rows);cv=pd.DataFrame(cov_rows);truth=pd.DataFrame(truth_rows)
    injected=truth[~truth.clean_control].copy();clean=truth[truth.clean_control].copy()
    nuisance=['n_qubits','topology_class','reference_layers','context','shots','strength_bin','partition','sampling_stratum']
    associations={col:cramers_v(injected.mechanism,injected[col]) for col in nuisance}
    shot_bin_v=cramers_v(injected.shots,injected.strength_bin)
    counts_by_split=frame.groupby(['partition','sampling_stratum']).size().rename('families').reset_index()
    coverage=pd.crosstab(frame.n_qubits,frame.topology_class)
    context=pd.crosstab(frame.topology_class,frame.injection_context_class)
    distributions={k:dict(Counter(str(v) for v in frame[k])) for k in ['n_qubits','topology_class','reference_layers','injection_context_class']}
    dev=frame[frame.partition!='TEST']
    stress_fraction=float(np.mean([any(g['min_norm']<.30 or g['smin']<.15 or g['max_cos']>.98 for g in f['geometry']) for f in families if f['partition']!='TEST' and f['sampling_stratum']=='representative']))
    bootdf=pd.DataFrame(bootstrap_rows)
    family_intervals={str(b):{name:cluster_interval(r[r.budget==b],name) for name in ['matched_sd_mean','coordinate_sd_mean']} for b in c.BUDGETS}
    summary={'cohort_identity':manifest['cohort_identity'],'status':manifest['status'],'model_evaluated':False,
        'test_labels_accessed':False,'test_array_files_opened_for_eda':0,
        'structural_distributions_all_partitions':distributions,'split_stratum_counts':counts_by_split.to_dict(orient='records'),
        'mechanism_counts_development':dict(Counter(injected.mechanism)),
        'clean_development_controls':len(clean),'development_mechanism_nuisance_cramers_v':associations,
        'development_shot_strength_bin_cramers_v':shot_bin_v,
        'representative_families_meeting_stress_rule_fraction':stress_fraction,
        'development_screen_attempts':len(screen),'development_rejection_reasons':dict(Counter(x['reason'] for x in screen if not x['accepted'])),
        'counts_summary_by_budget':{str(b):{'zero_fraction':quantiles(r[r.budget==b].count_zero_fraction)} for b in c.BUDGETS},
        'jacobian_uncertainty':{str(b):{'trace_ratio':quantiles(cv[cv.budget==b].trace_ratio_empirical_over_estimated),
            'approx95_mean_J_coverage':float(cv[cv.budget==b].approx95_mean_j_coverage.mean()),
            'mean_absolute_bias':quantiles(cv[cv.budget==b].mean_absolute_jacobian_bias)} for b in c.BUDGETS},
        'bootstrap_sample_roots':len(bootstrap_rows),'bootstrap_replicates':64,
        'bootstrap_coordinate_trace_ratio':quantiles(bootdf.coordinate_trace_ratio),
        'bootstrap_matched_trace_ratio':quantiles(bootdf.matched_trace_ratio),
        'bootstrap_max_coordinate_matched_correlation':quantiles(bootdf.max_abs_coordinate_matched_correlation),
        'minimum_analytic_covariance_eigenvalue':max_cov_psd_error,'minimum_bootstrap_covariance_eigenvalue':max_boot_psd_error,
        'family_cluster_intervals':family_intervals,'unverified_exclusions':manifest['unverified_exclusions'],
        'test_generation_truth_protected':True,'inherited_structural_proxy':'First RY gate acts on the intended affected qubit under unchanged Step14 grammar; not supplied as explicit hidden metadata.'}
    c.save_json(output/'eda_summary.json',summary);c.save_json(output/'joint_bootstrap_covariances.json',bootstrap_rows)
    r.to_csv(output/'root_geometry_summary.csv',index=False);cv.to_csv(output/'uncertainty_calibration.csv',index=False)
    counts_by_split.to_csv(output/'split_counts.csv',index=False)
    # Coverage: nominal grammar classes, not claims of distinct realized graphs.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    a=axes[0].imshow(coverage.to_numpy(),cmap='Blues');fig.colorbar(a,ax=axes[0],label='Families')
    axes[0].set_xticks(range(len(coverage.columns)),coverage.columns,rotation=40,ha='right');axes[0].set_yticks(range(len(coverage.index)),coverage.index)
    axes[0].set_ylabel('Qubits');axes[0].set_title('Nominal topology × qubit coverage (structural only)')
    for i in range(coverage.shape[0]):
        for j in range(coverage.shape[1]):axes[0].text(j,i,str(coverage.iloc[i,j]),ha='center',va='center')
    pd.crosstab(frame.partition,frame.sampling_stratum).plot.bar(stacked=True,color=[COLORS[k] for k in sorted(COLORS)],ax=axes[1])
    axes[1].set_ylabel('Families');axes[1].set_title('Fixed family split and sampling strata');axes[1].tick_params(axis='x',rotation=0)
    fig.tight_layout();fig.savefig(figdir/'01_structure.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    pd.Series(gate_counts).sort_index().plot.bar(ax=axes[0],color='#235c88');axes[0].set_ylabel('Operations');axes[0].set_title('Development clean-circuit gates')
    pd.crosstab(injected.strength_bin,injected.shots).plot.bar(ax=axes[1]);axes[1].set_title('Development strength bins × diagnostic shots');axes[1].set_xlabel('Strength bin');axes[1].set_ylabel('Observations')
    fig.tight_layout();fig.savefig(figdir/'02_design_balance.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    metrics=[('median_column_norm','Median Jacobian column norm'),('median_smin','Median normalized-frame smallest singular value'),('median_condition','Median normalized-frame condition'),('matched_sd_mean','Mean matched-response SD across draws')]
    for ax,(metric,label) in zip(axes.flat,metrics):
        for j,(stratum,color) in enumerate(COLORS.items()):
            a=r[r.sampling_stratum==stratum]
            groups=[a[a.budget==b][metric].to_numpy() for b in c.BUDGETS]
            ax.boxplot(groups,positions=np.arange(3)+j*.22-.11,widths=.18,showfliers=False,patch_artist=True,
                boxprops={'facecolor':color,'alpha':.65},medianprops={'color':'black'})
            ax.plot([],[],color=color,label=stratum)
        ax.set_xticks(range(3),['N','2N','4N']);ax.set_ylabel(label);ax.legend(fontsize=7)
        if metric!='median_smin':ax.set_yscale('log')
    fig.suptitle('Development geometry: candidate/draw medians within each root; no TEST geometry')
    fig.tight_layout();fig.savefig(figdir/'03_geometry.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for b,color in zip(c.BUDGETS,['#235c88','#3d9e91','#d9832e']):
        a=cv[cv.budget==b];axes[0].scatter(a.observed_estimated_trace,a.empirical_trace,s=8,alpha=.25,label=str(b),c=color)
    lo=min(cv.observed_estimated_trace.min(),cv.empirical_trace.min());hi=max(cv.observed_estimated_trace.max(),cv.empirical_trace.max())
    axes[0].plot([lo,hi],[lo,hi],'k--',lw=1);axes[0].set(xscale='log',yscale='log',xlabel='Observed-count analytic trace',ylabel='Empirical trace over 16 draws');axes[0].legend(title='Shots')
    axes[1].hist(np.log10(np.maximum(bootdf.coordinate_trace_ratio,1e-12)),bins=30,alpha=.6,label='Coordinates');axes[1].hist(np.log10(np.maximum(bootdf.matched_trace_ratio,1e-12)),bins=30,alpha=.6,label='Matched responses')
    axes[1].axvline(0,color='black',linestyle='--');axes[1].set(xlabel='log10(empirical variance / bootstrap variance)',ylabel='Development audit roots');axes[1].legend()
    fig.tight_layout();fig.savefig(figdir/'04_uncertainty.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));correlation=np.mean(bootcorr,axis=0)
    im=ax.imshow(correlation,cmap='RdBu_r',vmin=-1,vmax=1);fig.colorbar(im,ax=ax,label='Mean per-root bootstrap correlation')
    labs=['c_Z','c_X','c_Y','m_Z','m_X','m_Y'];ax.set_xticks(range(6),labs);ax.set_yticks(range(6),labs)
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{correlation[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_title('Joint coordinate / matched-response uncertainty (N)');fig.tight_layout();fig.savefig(figdir/'05_joint_covariance.png');plt.close(fig)
    geom_table=r.groupby(['sampling_stratum','budget'])[['median_column_norm','median_smin','median_condition','coordinate_sd_mean','matched_sd_mean']].median().reset_index()
    uc=cv.groupby('budget')[['trace_ratio_empirical_over_estimated','approx95_mean_j_coverage','mean_absolute_jacobian_bias']].median().reset_index()
    strength_summary=injected.groupby('strength_bin').strength.agg(['min','median','max','count']).reset_index()
    failures=pd.Series(summary['development_rejection_reasons'],name='attempts').rename_axis('reason').reset_index()
    lines=[
      '# Finite-shot cohort: exploratory data analysis',
      f"Cohort identity: `{manifest['cohort_identity']}`. Construction status: **{manifest['status']}**.",
      'This is a descriptive data-quality report. No frozen-network inference, accuracy measurement, remedy selection, or TEST outcome analysis was performed. All measurement/geometry plots use FIT and SELECTION only. TEST contributes structural counts and integrity checks in the separate audit.',
      '## Sampling, coverage and denominator',md_table(counts_by_split),
      f"Total: {len(families)} families, {len(roots)} clean roots and {len(roots)*13} diagnostic observations. Each root shares its calibration across its 13 observations. Draws and candidates are dependent descendants; they are not independent circuit samples.",
      '![Structure](figures/01_structure.png)',
      'Qubit × nominal topology counts:',coverage.to_markdown(),
      'Topology × injection context counts (structural metadata):',context.to_markdown(),
      'The six nominal topology labels are grammar choices, not six guaranteed graph-isomorphism classes. At two qubits they share one edge. Topology/context overlap with historical data is intentional; complete ordered skeleton duplication is excluded.',
      '## Acquisition and experimental balance',
      f"Each root has {manifest['draws_per_budget']} independent calibration realizations at each of N=4096, 2N=8192 and 4N=16384 **per axis/sign/basis call**. There are 18 calls per candidate per draw. Diagnostic shots remain 512/1024/2048/4096; reference and observed samples are independent. Diagnostics are held fixed across calibration draws, so this study isolates calibration randomness.",
      '![Balance](figures/02_design_balance.png)',
      'Development injected strengths:',md_table(strength_summary),
      'Development mechanism counts: '+json.dumps(summary['mechanism_counts_development'])+f"; clean controls: {len(clean)}.",
      'Mechanism/nuisance Cramér V (zero is exact fully crossed balance, not statistical independence of all continuous variables):',
      md_table(pd.DataFrame([{'nuisance':k,'Cramers_V':v} for k,v in associations.items()])),
      f"Shot/strength-bin Cramér V = {shot_bin_v:.5f}. A root-specific random permutation balances shot levels within each root; random finite-cohort residual associations are reported, not corrected after inspection.",
      '## Admission and rejected screening population',md_table(failures),
      f"Development screening attempts = {len(screen)}. Representative families that also meet the stress geometry rule: {stress_fraction:.1%}. Sampling strata therefore are not disjoint difficulty classes. Failed original admission is recorded as a screening population, not silently dropped from the target population. TEST screening details remain sealed.",
      'Original admission is a mechanism-separation test at strength 0.12. It does not prove identifiability at every smaller strength or certify matrix full rank. Null local derivatives and non-identifiable finite mechanisms are different properties.',
      '## Geometry and numerical behavior','![Geometry](figures/03_geometry.png)',md_table(geom_table),
      'Rows above summarize medians over root summaries. Increased shots need not monotonically improve nonlinear coordinate or normalized-direction features, particularly near weak/null directions. Boxplots suppress plotting outliers for readability; all observations remain in CSV summaries and all quantile ranges remain in JSON. No EDA-driven filtering was applied.',
      '## Count uncertainty, bias and covariance validation','![Uncertainty](figures/04_uncertainty.png)',md_table(uc),
      'The analytic covariance is estimated from observed counts only, with the N−1 unbiased correction and correlated statistics within a basis. Calibration plus/minus and basis blocks are acquired independently. Empirical trace uses 16 calibration draws of the first ordered candidate; that trace is noisy. The nominal 95% mean-J coverage uses normal intervals against the transient simulator central-difference target, excludes zero-variance cells, and is descriptive, not a formal coverage guarantee. Finite-difference truncation relative to the infinitesimal derivative is not estimated here.',
      f"Analytic covariance minimum numerical eigenvalue: {max_cov_psd_error:.3g}; joint bootstrap minimum: {max_boot_psd_error:.3g}. Tiny negative values reflect floating-point roundoff, not silent covariance repair.",
      f"Joint uncertainty: {len(bootstrap_rows)} development roots, one first-ordered candidate and opaque first observation each, N draw zero, 64 observed-count bootstrap replicates. Diagnostic evidence is held fixed. Empirical/bootstrapped coordinate trace ratio quantiles: {json.dumps(summary['bootstrap_coordinate_trace_ratio'])}. Matched trace ratio quantiles: {json.dumps(summary['bootstrap_matched_trace_ratio'])}.",
      '![Joint covariance](figures/05_joint_covariance.png)',
      'Coordinate and matched-response estimates arise from the same noisy response frame and must not be treated as independent. A bootstrap can disagree strongly with empirical variation at sparse/null boundaries. These results validate calculability and expose calibration limitations; they do not validate an uncertainty-based stabilization remedy. All six-by-six per-root matrices are retained in `joint_bootstrap_covariances.json`.',
      '## Family-cluster descriptive intervals',
      md_table(pd.DataFrame([{'budget':b,'metric':metric,**v} for b,m in family_intervals.items() for metric,v in m.items()])),
      'Intervals resample equal-weight family means, 2,000 bootstrap replicates with a fixed reporting seed. They quantify descriptive uncertainty within this generated cohort, not population guarantees or remedy efficacy.',
      '## Integrity, scientific limitations and verdict',
      'The separate independent audit checks hashes, shot mass, split isolation, blind-field allowlists and source-reconstructed measurements/features. Its independent simulator replay uses the same Qiskit library and is not an independent hardware replication.',
      '**Inherited structural proxy:** the original generator puts its first RY on the intended affected qubit. This association is present in the known clean graph. We preserve it to retain the task; removal would change the experimental distribution. Explicit true-location fields are absent from estimator inputs, but this corpus cannot support a claim of location-uninformative circuit structure.',
      '**Unresolved historical exclusion coverage:** '+', '.join(manifest['unverified_exclusions'])+'. No protected reserve was opened to close this gap. The cohort remains quarantined and must not select stabilization methods until these exclusion checks are completed through authorized blind identities.',
      'No claim is made about prediction accuracy, improvement, optimal shot count, or hardware transfer. Raw arrays, recipe hashes, code, source bindings and versioned archive references are documented separately. TEST labels and recipes are encrypted; this report never decrypted them.',
      '## Reproduction data products',
      '`eda_summary.json`, `root_geometry_summary.csv`, `uncertainty_calibration.csv`, `joint_bootstrap_covariances.json`, `split_counts.csv`, and five figures are generated by `scripts/cohort/eda.py`. Full ranges, covariance outliers and stratum tables remain in those machine-readable products.'
    ]
    (output/'EDA_REPORT.md').write_text((chr(10)*2).join(lines)+chr(10))
    print(json.dumps({'cohort_identity':manifest['cohort_identity'],'eda_dev_roots':len(r)//3,'bootstrap_roots':len(bootstrap_rows),'test_array_files_opened':0}),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--product',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.product,a.output)
