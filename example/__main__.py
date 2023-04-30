import argparse
import os
import pprint
from dataclasses import dataclass

from fyst import Cel, Row, Table

from src.cli import Cli
from src.serialize import SerializableData

from .model import ModelParam
from .train import Trainer, TrainParam


# define cli params
@dataclass
class CliParam(SerializableData):
    dir: str
    force: bool = False


# define cli parser
cli = Cli(tuple[CliParam, ModelParam, TrainParam])

# table = Table()

table = Table(padding=(0, 0, 2, 0), border=0)
for group in cli._parser._action_groups:
    if len(group._group_actions) == 0:
        continue
    # print(group.title)
    table.append(Row(Cel(group.title, span=(4, 1), padding=(0, 0, 0, 1))))
    for action in group._group_actions:
        default = action.default or ""
        if action.default == argparse.SUPPRESS:
            default = ""
        table.append(
            Row("  ", ", ".join(action.option_strings), action.metavar or "", action.help or "")
        )
    table.append(Row(Cel()))
        # print(", ".join(action.option_strings), action.metavar, default, action.help or "", sep="\t")
    # print(table)
print(table)
# print(*cli._parser._action_groups[2]._group_actions, sep="\n")
exit()

# parse
cparam, mparam, tparam = cli.parse()

# exit on existing folder
if os.path.exists(cparam.dir) and not cparam.force:
    print(f"dir {cparam.dir} already exists. Run with --force True to continue.")
    exit(0)

# make missing dirs
os.makedirs(cparam.dir, exist_ok=cparam.force)

# print params
print("\nModel:")
pprint.pp(mparam.serialize(mparam))

print("\nTrain:")
pprint.pp(tparam.serialize(tparam))

# train
trainer = Trainer(mparam, tparam)
trainer.train(cparam.dir)
trainer.save(cparam.dir)

print("Done!")
