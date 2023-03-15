# silent_tools
A bunch of shell utilities for dealing the silent files

```bash
# make a silent file
silentfrompdbs *.pdb > my.silent

# ask what's in a silent file
silentls my.silent  

# ask how many things are in a silent file
silentls my.silent | wc -l   

# extract all pdbs from a silent file
silentextract my.silent   

# extract the first 10 pdbs from a silent file
silentls my.silent | head -n 10 | silentextractspecific my.silent    

# extract a random 10 pdbs from a silent file
silentls my.silent | shuf | head -n 10 | silentextractspecific my.silent  

# extract a specific pdb from a silent file
silentextractspecific my.silent name_of_pdb_0001

# produce a scorefile from a silent file
silentscorefile my.silent   

# combine silent files
cat 1.silent 2.silent 3.silent > my.silent  

# ensure all pdbs in silent file have unique names
silentls my.silent | silentrename my.silent > uniq.silent  

# remove _0001 from all the names in a silent file
silentls my.silent | sed 's/_0001$//g' | silentrename my.silent > renamed.silent

# make a new silent file with the first 10 pdbs
silentls my.silent | head -n 10 | silentslice my.silent > new.silent  

# get all sequence from a silent file
silentsequence my.silent > my.seq

# split a silent file into groups of 100
silentsplit my.silent 100
```

# Installation

Installation is simple. Simply clone the repository and add it to your path.
```bash
cd ~
mkdir software
cd software
git clone https://github.com/bcov77/silent_tools
cd silent_tools
echo "PATH=\$PATH:$(pwd)"
```
These commands will print a line that looks like `PATH=...`. Add this to your .bashrc

Additionally, `silentfrompdbs` and `silentextract` require Rosetta. If you have already installed PyRosetta in your
default python, then you're already good to go. Otherwise you'll need to make some symlinks.
```bash
cd ~/software/silent_tools
ln -s /my/path/to/rosetta/main/source/bin/score_jd2 .
ln -s /my/path/to/rosetta/main/source/bin/extract_pdbs .
```

# Refresher on Silent Files

There are many types of silent files, but there is only one type that should ever be used. <b>Binary silent files</b> are
the only type that should ever be used.

There are precisely 2 reasons to use silent files:

1. All of your outputs are in one file (instead of 1M pdbs...)

2. Silent files load 10X faster than pdbs.

Both of these points are huge. You won't clog up filesystems and you won't have to transfer absurd numbers of files between
systems. Additionally, the faster load times make a huge difference when you forget to run a filter and have to
load all 1M of your outputs just to spend 0.1 second analyzing each one.



To tell Rosetta to output to a silent file, use this:
```bash
rosetta_scripts ... -out:file:silent_struct_type binary -out:file:silent out.silent
```
This will produce one silent file with all your pdbs.

If you produce many silent files, it is almost always easier to combine them into one giant one.
```bash
cat runs/*/out.silent > combined.silent
```

To tell Rosetta to read from a silent file, use this:
```bash
rosetta_scripts ... -in:file:silent_struct_type binary -in:file:silent in.silent
```

If you only want to work with a subset of the designs in a silent file, the easy solution is to do this:
```bash
rosetta_scripts ... -in:file:silent_struct_type binary -in:file:silent in.silent -tags tag1 tag2 tag3 tag4
OR
rosetta_scripts ... -in:file:silent_struct_type binary -in:file:silent in.silent -tagfile tags.list
```

However, a better way is this:
```bash
silentslice in.silent tag1 tag2 tag3 tag4 > tmp.silent
rosetta_scripts ... -in:file:silent_struct_type binary -in:file:silent tmp.silent
OR
cat tags.list | silentslice in.silent > tmp.silent
rosetta_scripts ... -in:file:silent_struct_type binary -in:file:silent tmp.silent
```
This way, Rosetta doesn't need to read the entire silent file. You only give it the bare minimum. `silentsplit` is 
useful here too.

# But I don't want to extract silent files

Extracting silent files is so easy though!
```bash
silentextract my.silent
```

But I only want to look at one silent pdb!
```bash
silentextractspecific my.silent that_one_I_want_to_look_at
```

What about a list of pdbs?
```bash
cat list_of_tags.list | silentextractspecific my.silent
```

---

I would argue to you that silent files provide a more natural way to work with pdbs. Consider a workflow where we make
a bunch of outputs and then filter by column 10 in the scorefile.

With silent files:
```bash
cat runs/*/out.silent > combined.silent
silentscorefile combined.silent
cat combined.sc | awk '{if ($10 > 100) {print $NF}}' | silentextractspecific combined.silent
```

Without silent files:
```bash
find runs/ -name '*.pdb' > pdbs.list
cat runs/*/*.sc > combined.sc
cat combined.sc | awk '{if ($10 > 100) {print $NF ".pdb"}}' | grep -f - pdbs.list > good_paths.list
cat good_paths.list | xargs cp -t .
```

Both methods work, but I hope you agree that the one with silent files is cleaner. `runs/` may be safely deleted in the
silent file case leaving you with a single file representing your whole run.
