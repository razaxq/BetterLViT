"""Reproduce AST checks against the archived R2 evaluator and frozen data loaders."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
DOCS=HERE.parents[2]
OLD=DOCS/'repro_archive/20260910/recipe_test/evaluate_test.py'
NEW=HERE/'evaluate_test.py'
COMMITS=dict(r2='9eca26de5b301099805530edbf5a1a8718bea662',s1='1f7edb7858345b6b3ef0736291bb5cf92a2f712f',s2='7defc637e36974fabd0f47763275c90f617e5f53')


def node(tree,name):
    return next(n for n in tree.body if getattr(n,'name',None)==name)


def frozen(label,path):
    repo=Path('D:/BetterLViT')/('recipe_r2_work' if label=='r2' else f'visual_aux_{label}_work')
    return ast.parse(subprocess.check_output(['git','-C',str(repo),'show',COMMITS[label]+':'+path],encoding='utf-8'))


if __name__=='__main__':
    old=ast.parse(OLD.read_text());new=ast.parse(NEW.read_text())
    functions=('mask_boundary','boundary_f1','image_frequency_scores','build_model','test_loader')
    for name in functions:
        assert ast.dump(node(old,name))==ast.dump(node(new,name)),name
    a=next(n for n in node(old,'main').body if isinstance(n,ast.With))
    b=next(n for n in node(new,'main').body if isinstance(n,ast.With))
    assert ast.dump(a)==ast.dump(b)
    comparisons={}
    for label in ('s1','s2'):
        for path,names in (('Load_Dataset.py',('ValGenerator','ImageToImage2D')),('utils.py',('read_text',))):
            baseline=frozen('r2',path);candidate=frozen(label,path)
            for name in names:
                assert ast.dump(node(baseline,name))==ast.dump(node(candidate,name)),(label,path,name)
                comparisons[f'{label}:{path}:{name}']='identical'
    result=dict(r2_evaluator_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),
                evaluator_sha256=hashlib.sha256(NEW.read_bytes()).hexdigest(),
                unchanged_functions=list(functions),inference_loop_ast_identical=True,
                loader_ast_comparison=comparisons,verified_training_source_commits=COMMITS)
    (HERE/'evaluator_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
