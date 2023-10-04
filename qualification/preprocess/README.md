## Preprocess


<details><summary>1. Convert the original XML files into YOLO TXT format.</summary>

```bash
$ python xml2txt.py
```

</details>


<details><summary>2. Organize in a folder structure that is conducive to training.</summary>

```bash
$ python folderStructure.py
```

</details>


<details><summary>3. Resplit the dataset into better validation-training ratio.</summary>

```bash
$ python resplit.py
```

</details>


<details><summary>4. Implement fisheye distortion.</summary>

```bash
$ ./fisheye
```

</details>


<details><summary>5. Implement data augmentation.</summary>

```bash
$ python data_aug.py
$ python data_aug_2.py
```

</details>


<details><summary>6. Statistics on annotations.</summary>

```bash
$ python statistics.py
```

</details>
