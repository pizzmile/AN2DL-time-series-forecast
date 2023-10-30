# AN2DL - Project 2

## Directory tree
- `data` - *Holds the dataset to train and test models*
- `models` - *Holds the settings and the output of the models*
  - `img` - *Holds the images of models' architectures*
  - `logs` - *Holds the logs of models' training*
  - `settings` - *Holds the configurations to build and train models*
  - `tests` - *Holds the results of tests on models*
- `scripts` - *Holds the modules used to make this app works*

## Configuration
Here are the available constructors for *.yaml* files:
| Name | Identifier | Effect |
|------|------|-----|
| **Join paths** |`!joinPath`| Join a list of paths together through _os.path.join_|
| **Join** | `!join` | Join a list of _str_|
| **To tuple** | `!tuple` | Convert a list of two elements into a tuple|
| **To float** | `!float` | Convert a float in scientific notation into a standard float|
| **Divide** | `!divide` | Return the division of the elements of a list of two numbers|
| **Map to list** | `!mapToList` | Create a list from the values of a map|
| **Layer** | `!layer` | Create a tensorflow layer object given its _type_ and _params_|
| **Sequence** | `!sequence` | Create a tensorflow Sequence object given its _params_|
| **Network** | `!network` | Load and set up a pre-trained network from tensorflow given its _type_ and _params_|
| **Optimizer** | `!optimizer` | Create a tensorflow optimizer object given its _type_ and _params_|
| **Initializer** | `!initializer` | Create a tensorflow initializer object given its _type_ and _params_|
| **Loss** | `!loss` | Create a tensorflow loss object given its _type_ and _params_|

**NOTE** 
For each tensorflow object, if it is not defined yet, then a definition inside the dicts (in the _tf_ module) 
is needed in order to compile


### Compiler configuration
The `config.yaml` file in the root directory is used to define and set up the directory tree of the 
`main.py` file of the compiler.

### Model configuration
Models' configurations are stored in `./models/settings/` folder.
