from utils import (
    Medbullets_op4,
    Medbullets_op5,
    MedXpertQA,
    MMMU,
    OmniMedVQA,
    PATH_VQA,
    PMC_VQA,
    SLAKE,
    SuperGPQA,
    VQA_RAD,
    HealthBench,
    PubMedQA,
    MedMCQA,
    MedQA_USMLE,
    MMLU,
    CMB,
    CMExam,
    MedQA_MCMLE,
    CMMLU,
    IU_XRAY,
    CheXpert_Plus,
    MIMIC_CXR,
    MedFrameQA,
    OmniBrainBench
    )

#eval_MedQA_USMLE
def prepare_benchmark(model,eval_dataset,eval_dataset_path,eval_output_path):
    # Hallu,Geometry3k
    supported_dataset = ["MMMU-Medical-test","MMMU-Medical-val","PATH_VQA","PMC_VQA","VQA_RAD","SLAKE","MedQA_USMLE","MedMCQA","PubMedQA","OmniMedVQA","Medbullets_op4","Medbullets_op5","MedXpertQA-Text","MedXpertQA-MM","SuperGPQA""HealthBench","IU_XRAY","CheXpert_Plus","MIMIC_CXR","CMB","CMExam","CMMLU","MedQA_MCMLE","MedFrameQA", "OmniBrainBench", "OmniBrainBench-Open"]

    if eval_dataset in ["MMMU-Medical-test", "MMMU-Medical-val"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"MMMU")
        _ , subset , split = eval_dataset.split("-")
        dataset = MMMU(model,eval_dataset_path,eval_output_path,split,subset)

    elif eval_dataset == "PATH_VQA":
        dataset = PATH_VQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "PMC_VQA":
        dataset = PMC_VQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "VQA_RAD":
        dataset = VQA_RAD(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "SLAKE":
        dataset = SLAKE(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "OmniMedVQA":
        dataset = OmniMedVQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "Medbullets_op4":
        dataset = Medbullets_op4(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "Medbullets_op5":
        dataset = Medbullets_op5(model,eval_dataset_path,eval_output_path)

    elif eval_dataset in ["MedXpertQA-Text","MedXpertQA-MM"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"MedXpertQA")
        _,split = eval_dataset.split("-")
        dataset = MedXpertQA(model,eval_dataset_path,eval_output_path,split)

    elif eval_dataset == "SuperGPQA":
        dataset = SuperGPQA(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset == "HealthBench":
        dataset =HealthBench(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset == "PubMedQA":
        dataset = PubMedQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "MedMCQA":
        dataset = MedMCQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "MedQA_USMLE":
        dataset = MedQA_USMLE(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset in ["MMLU-medical","MMLU-all"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"MMLU")
        _,subject = eval_dataset.split("-")
        dataset = MMLU(model,eval_dataset_path,eval_output_path,subject)
    
    elif eval_dataset == "CMB":
        dataset = CMB(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset == "CMExam":
        dataset = CMExam(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "MedQA_MCMLE":
        dataset = MedQA_MCMLE(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "CMMLU":
        dataset = CMMLU(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset == "IU_XRAY":
        dataset = IU_XRAY(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "CheXpert_Plus":
        dataset = CheXpert_Plus(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "MIMIC_CXR":
        dataset = MIMIC_CXR(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset == "MedFrameQA":
        dataset = MedFrameQA(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset.startswith("OmniBrainBench"):
        if eval_dataset == "OmniBrainBench-Open":
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"OmniBrainBench")
            dataset = OmniBrainBench(model, eval_dataset_path, eval_output_path, openset=True)
        else:   
            dataset = OmniBrainBench(model,eval_dataset_path,eval_output_path, openset=False)
    else:
        print(f"unknown eval dataset {eval_dataset}, we only support {supported_dataset}")
        dataset = None

    return dataset

if __name__ == '__main__':
    pass    