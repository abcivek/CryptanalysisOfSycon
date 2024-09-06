This is the Sycon adaptation of linear trails tools presented by Dobraunig, C., Eichlseder, M., & Mendel, F.

Instructions
Get the original tool: git clone https://github.com/iaikkrypto/lineartrails.git
Go to the lineartrails folder.
Copy the content in this repo's "target" folder to the "target" folder in the lineartrails.
Copy the content in this repo's "example" folder to the "example" folder in the lineartrails.
Go to the "tool" folder and make some editing:
	Edit "permutation_list.cpp" like this:
 		Add this line before "assert":
    		if (name.compare("sycon") == 0)
    		return new SyconPermutation(rounds);

	Edit "permutation_list.h" like this:
 		Add this line: 
 			#include "sycon_permutation.h"

git submodule init && git submodule update
make

Run: ./lin -I 10 -S 2 -i examples/sycon_3_rounds_typeI.xml