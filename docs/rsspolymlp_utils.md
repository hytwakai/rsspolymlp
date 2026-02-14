Using rsspolymlp-utils --struct_matcher
```shell
rsspolymlp-utils --struct_matcher --poscar ./POSCAR* --output_file result.yaml
```

Using rsspolymlp-utils --geometry_opt
```shell
rsspolymlp-utils --geometry_opt --poscar ./POSCAR* --pot polymlp.yaml --pressure 0
# --symmetry: the optimization is comducted with using symmetry constraints
```
