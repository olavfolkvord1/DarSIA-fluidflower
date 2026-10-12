"""Phone-readable scientific comparison of the six optical calibration branches."""
from pathlib import Path
import numpy as np
from darsia.fluidflower.tools import phase_studio_native_mass as io


LABELS = {
    'none_cc_off':'No illumination / CC off', 'none_cc_on':'No illumination / CC on',
    'historical_cc_off':'Historical / CC off', 'historical_cc_on':'Historical / CC on',
    'facies_cc_off':'Facies / CC off', 'facies_cc_on':'Facies / CC on',
}


def equal_run_error(rows_by_run, doses):
    """Every experiment has equal weight; zero-dose controls excluded explicitly."""
    records={}
    for name,rows in rows_by_run.items():
        rows=[row for row in rows if row['nominal_injected_mass_g']>0]
        if not rows:
            raise ValueError('No positive-dose observations for '+name)
        error=np.asarray([row['total_mass_g']-row['nominal_injected_mass_g'] for row in rows])
        records[name]=dict(n=len(rows), nmae=float(np.abs(error).mean()/doses[name]),
                           nrmse=float(np.sqrt(np.mean(error**2))/doses[name]))
    return dict(nmae=float(np.mean([x['nmae'] for x in records.values()])),
                nrmse=float(np.mean([x['nrmse'] for x in records.values()])),runs=records)


def build_report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    request=io.read(root/'request_binding.json')
    coverage=io.read(root/'coverage.json')
    names=request['runs']
    series={'ffum':'FF_UM','ffac':'FF_AC','ffbm':'FF_BM'}.get(
        str(request.get('dataset','ffum')).lower(),str(request.get('dataset','series')).upper())
    result={}
    fits_by_variant={}
    for variant in coverage['accepted']:
        key=variant['photometry_variant_id']
        fits_path=root/key/'fits.json'
        if not fits_path.is_file():
            continue
        fits=io.read(fits_path)
        fits_by_variant[key]=fits
        cv=io.read(root/key/'cross_validation.json')
        summaries=io.read(root/key/'summaries.json')
        doses={name:float(summaries[name][0]['expected_final_injected_mass_g']) for name in names}
        data=dict(variant=variant,shared_training=equal_run_error(fits['shared']['predictions'],doses),
                  individual_training=equal_run_error({name:fits['individual'][name]['predictions'][name] for name in names},doses))
        for mode in ('interleaved','chronological'):
            for scope in ('shared','individual'):
                data[scope+'_'+mode]=equal_run_error({name:fold['rows'] for name,fold in cv['time_folds'][mode][scope].items()},doses)
        data['whole_run_holdout']=equal_run_error({name:fold['rows'] for name,fold in cv['whole_run_holdout'].items()},doses)
        data['shared_parameters']=fits['shared']['shared_parameters']
        data['shared_warnings']=fits['shared'].get('warnings',[])
        data['optimizer']=fits['shared'].get('diagnostics',{}).get('optimization',{})
        data['reference_validity']={name:io.read(root/key/'preparation'/name/'prepared.json')['reference_diagnostics']['photometric_valid_fraction'] for name in names}
        result[key]=data
    if not result:
        return
    folder=root/'comparison'
    folder.mkdir(exist_ok=True)
    io.write(folder/'comparison.json',dict(method='equal-run normalized mass error; fixed full protocol dose per run',
        primary_metric='shared interleaved held-out NMAE; compare chronological and held-experiment errors separately',
        variants=result,rejected=coverage['rejected'],
        limitations=['Inverse mass fit is not independent local concentration validation.',
                     'Same phase maps, geometry and physical arrays; fresh raw-derived RGB for every optical branch.',
                     'Clearwater source and physical assumptions remain bound per experiment.',
                     'Do not select solely on training mass fit or chart self-match.']))
    order=[key for key in LABELS if key in result]
    colors={'none':'#276fbf','historical':'#c95f21','facies':'#31815b'}
    fig,axes=plt.subplots(3,1,figsize=(9,13),constrained_layout=True)
    for ax,metric,title in zip(axes,['shared_training','shared_interleaved','whole_run_holdout'],
                             ['Shared fit: training error','Shared fit: held-out times','Transfer: held-out experiments']):
        values=[100*result[key][metric]['nmae'] for key in order]
        bars=ax.barh([LABELS[key] for key in order],values,
            color=[colors[key.split('_cc_')[0]] for key in order],
            alpha=.85)
        for bar,key,value in zip(bars,order,values):
            if key.endswith('off'):bar.set_hatch('//')
            ax.text(value+.08,bar.get_y()+bar.get_height()/2,f'{value:.2f}%',va='center',fontsize=12)
        ax.invert_yaxis();ax.set_title(title,fontsize=16);ax.set_xlabel('Mean absolute error / injected dose (%)')
        ax.set_xlim(0,max(values)*1.22+0.5);ax.grid(axis='x',alpha=.2);ax.tick_params(labelsize=11)
    frame_count=sum(request['expected_frame_counts'].values())
    fig.suptitle(f'{series} — illumination and ColorChecker\n{len(names)} experiments · {frame_count} fixed phase-review frames',fontsize=17)
    fig.savefig(folder/'comparison_mobile.png',dpi=150);plt.close(fig)
    for name in names:
        for scope in ('shared','individual'):
            fig,axes=plt.subplots(3,1,figsize=(8,12),sharex=True,constrained_layout=True)
            for key in order:
                fits=fits_by_variant[key]
                fit=fits['shared'] if scope=='shared' else fits['individual'][name]
                rows=fit['predictions'][name]
                time=[row['elapsed_h'] for row in rows]
                for ax,field in zip(axes,['total_mass_g','gas_mass_g','aqueous_mass_g']):
                    ax.plot(time,[row[field] for row in rows],
                        color=colors[key.split('_cc_')[0]],linestyle='-' if key.endswith('on') else '--',
                        label=LABELS[key],linewidth=1.8)
            axes[0].plot(time,[row['nominal_injected_mass_g'] for row in rows],color='black',linewidth=2.1,label='Injected CO2')
            for ax,title in zip(axes,['Total detected vs injected','Free gas','Dissolved CO2 (including water in gas regions)']):
                ax.set_title(title,fontsize=13);ax.set_ylabel('CO2 mass (g)');ax.grid(alpha=.25)
            axes[0].legend(fontsize=9,ncol=2);axes[-1].set_xlabel('Time (hours)')
            fig.suptitle(name.upper()+' · '+scope+' calibration\nSame phase maps and physical inputs',fontsize=16)
            fig.savefig(folder/(name+'_'+scope+'_mobile.png'),dpi=140);plt.close(fig)
    lines=[f'# {series}: photometric calibration variants','',
           f'All accepted variants use the same {len(names)} experiments and {frame_count} frames. Values below are equal-run NMAE, normalized by each full injection dose. The optimization budget and validation splits are identical.', '',
           '| Variant | Shared training | Held-out times | Late-time holdout | Held-out experiment |',
           '|---|---:|---:|---:|---:|']
    for key in order:
        row=result[key]
        lines.append('| '+LABELS[key]+' | '+' | '.join(f"{100*row[m]['nmae']:.3f}%" for m in
            ('shared_training','shared_interleaved','shared_chronological','whole_run_holdout'))+' |')
    lines+=['','Individual fits and every run are recorded in comparison.json and the portrait mass plots.',
            'The shared fit is not independent concentration truth. Review clipping, baseline-relative spatial artifacts, weak indicator responses, and the phase split alongside mass error.',
            'Clearwater sources and physical assumptions are recorded per experiment. Overflow use is recorded in the immutable request.',
            'Rejected branches: '+str(coverage['rejected'])]
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
