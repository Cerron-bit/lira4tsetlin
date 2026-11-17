# Copyright 2026 Yannick Fuchs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

{
  description = "Environment for running modern Tsetlin Machines.";

  inputs.nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-26.05";

  outputs = { nixpkgs, ... } @ inputs:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
        cudaVersion = "12";
      };
    };
  in
  {
    devShells.${system}.default = with pkgs; mkShell {
      buildInputs = [
        # python specific tools.
        virtualenv
        python312
        poetry
        # CUDA specific tools.
        cudaPackages.cudatoolkit
      ];

      # Env-variable needed for C-libs.
      LD_LIBRARY_PATH = lib.makeLibraryPath [
        # libs for cuda.
        cudaPackages.cuda_cudart
        cudaPackages.cudnn
        # libs for numpy.
        stdenv.cc.cc
        stdenv.cc.cc.lib
        zlib
      ];

      OMP_NUM_THREADS=10;

      shellHook = ''
        VENV_DIR="tsetlin_env"
        if [[ ! -d "$VENV_DIR" ]] then
          echo ">>>>>>>>>> MISSING VIRTUAL EVNIRONMENT DETECTED <<<<<<<<<<"
          virtualenv -p python3.12 "$VENV_DIR"
          source "$VENV_DIR"/bin/activate
          echo ">>>>>>>>>>> VIRTUAL ENVIRONMENT CREATED & LIVE <<<<<<<<<<<"
          poetry install
          cd pyTsetlinMachineParallel
          python3 setup.py build
          python3 setup.py install
          rm -r build pyTsetlinMachineParallel.egg-info
          cd ../
          export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.12/site-packages/libTM.cpython-312-x86_64-linux-gnu.so:$LD_LIBRARY_PATH"
        else
          echo ">>>>>>>>>>>>>>>> VIRTUAL ENVIRONMENT LIVE <<<<<<<<<<<<<<<<"
          source "$VENV_DIR"/bin/activate
        fi
        export CUDA_HOME="${cudaPackages.cudatoolkit}"
        export LD_LIBRARY_PATH="${cudaPackages.cudatoolkit}/lib:${cudaPackages.cuda_cudart}/lib:${cudaPackages.cudnn}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
        export PATH="${cudaPackages.cudatoolkit}/bin:$PATH"
        echo ">>>>>>>>>>>>>>>>>>>>> DETECTING GPU <<<<<<<<<<<<<<<<<<<<<<"
        python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 2>/dev/null | grep "PhysicalDevice"
        echo ">>>>>>>>>>>>>> CHECKING MODIFIED TM-MODULE <<<<<<<<<<<<<<<"
        python3 -c "from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D; TM = MultiClassConvolutionalTsetlinMachine2D(number_of_clauses=2000,T=50 * 100,s=5.0,patch_dim=(10, 10),weighted_clauses=True); print('Reachable:', TM.T == 5000)"
      '';
    };
  };
}
