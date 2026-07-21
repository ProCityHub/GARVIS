#!/usr/bin/env python3
"""Print the GARVIS anatomy-inspired software plan."""

from garvis.anatomical_architecture import AnatomicalHeartbeat
from garvis.anatomical_architecture.knowledge import anatomy_software_learning_pack

print(anatomy_software_learning_pack())
print()
print(AnatomicalHeartbeat().run("Design GARVIS as an integrated software organism.").summary)
