import os
import torch
import torchvision.transforms as t
import faiss       
import tqdm       
import numpy as np  
import albumentations as A  
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from dataset import EbayDataset, EbayDatasetSiamse
from models import FineTunedResnetCE, FineTunedResnetArc, FineTunedResnetSiamse
from utils import parse_config


def build_index(dataset, model, vor_cells):
    print("Loading dataset for building index")
    dl = DataLoader(dataset, batch_size=512, num_workers=4)

    index = faiss.index_factory(1000, f"IVF{vor_cells},PQ100")

    data_transformed = None
    for batch, *_ in tqdm.tqdm(dl):
        if data_transformed is None:
            data_transformed = model(batch)
        else:
            data_transformed = torch.cat((data_transformed, model(batch)), dim=0)
    
    data = data_transformed.detach().numpy()

    print("Training the index")
    index.train(data)
    print("Filling the index")
    index.add(data)
    return index


def build_index_on_disk(dataset, model, vor_cells, index_folder):
    print("Loading dataset for building index")
    batch_size = 256
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    index = faiss.index_factory(64, f"IVF{vor_cells // 4},PQ16")

    # train index
    print("Training index")
    train_samples = torch.stack([dataset[i][0] for i in range(batch_size)], dim=0)
    train_output = model(train_samples).detach().numpy()
    index.train(train_output)
    faiss.write_index(index, f"{index_folder}/trained.index")

    # add block indexes
    print("Adding data to index")

    num_batches = 0
    for i, (batch, *_) in tqdm.tqdm(enumerate(dl)):
        if len(batch) != batch_size:
            break
        output = model(batch).detach().numpy()
        index = faiss.read_index(f"{index_folder}/trained.index")
        index.add_with_ids(output, np.arange(i * batch_size, (i + 1) * batch_size).astype('int64'))
        faiss.write_index(index, f"{index_folder}/block_{i}.index")
        num_batches = i + 1
    
    block_fnames = [f"{index_folder}/block_{b}.index" for b in range(num_batches)]
    index = faiss.read_index(f"{index_folder}/trained.index")
    for i, fname in enumerate(block_fnames):
        index.merge_from(faiss.read_index(fname), i)
    faiss.write_index(index, f"{index_folder}/populated.index")
    index.nprobe = vor_cells
    return index


def search_index(index: faiss.IndexFlatL2, model, dataset, k=3):
    print("Loading dataset for searching index")
    dl = DataLoader(dataset, batch_size=128, num_workers=4)
    data_transformed = None
    for batch, *_ in tqdm.tqdm(dl):
        if data_transformed is None:
            data_transformed = model(batch)
        else:
            data_transformed = torch.cat((data_transformed, model(batch)), dim=0)
        break

    data = data_transformed.detach().numpy()
    print("Searching the index")
    _, nearest = index.search(data, k)
    return nearest  


def save_retrieval(from_dataset: EbayDataset, db_dataset: EbayDataset, model: torch.nn.Module, index: faiss.IndexFlatL2, out_folder, k=3):
    samples = []
    sample_classes = []
    for class_label in from_dataset.info_pd["super_class_id"].unique():
        samples.append(from_dataset.get_random_image_from_class(class_label))
        sample_classes.append(class_label)
    samples = torch.stack(tuple(samples), dim=0)

    transformed = model(samples).detach().numpy()
    _, nearest = index.search(transformed, k)

    for i, sample in enumerate(nearest):
        neighbours = [samples[i]]
        for neighbour_index in sample:
            image, *_ = db_dataset[neighbour_index]
            neighbours.append(image)
        grid = make_grid(neighbours, nrow=1)
        save_image(grid, f"{out_folder}/class_{sample_classes[i]}_example.png")


if __name__ == "__main__":
    config = parse_config("./config.yaml")
    train_config = config["train_data"]
    test_config = config["test_data"]

    # model = models.resnet18(pretrained=True)
    model = FineTunedResnetCE.load_from_checkpoint("checkpoints_cross-entropy\model_cross_entropy-epoch=04-val_loss=0.99.ckpt", config=config)
    train_ds = EbayDataset(train_config)
    test_ds = EbayDataset(test_config)
    vor_cells = 4*int(np.sqrt(512))
    index_out_folder = "index_fine_tuned_ce"
    saved_results = "saved/fine_tuned_ce"

    os.makedirs(index_out_folder, exist_ok=True)
    os.makedirs(saved_results, exist_ok=True)

    if os.path.exists(f"{index_out_folder}/populated.index"):
        index  = faiss.read_index(f"{index_out_folder}/populated.index")
    else:
        index = build_index_on_disk(train_ds, model, vor_cells, index_out_folder)

    index.nprobe = vor_cells

    save_retrieval(test_ds, train_ds, model, index, saved_results)
