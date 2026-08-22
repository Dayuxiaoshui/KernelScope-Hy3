from kernelscope.gpu_audit import ComputeProcess, unexpected_processes

def test_unexpected_processes_are_scoped_to_gpu_and_pid():
    rows = [ComputeProcess("gpu-a", 10, 100), ComputeProcess("gpu-a", 11, 200), ComputeProcess("gpu-b", 12, 300)]
    assert unexpected_processes("gpu-a", {10}, rows) == [ComputeProcess("gpu-a", 11, 200)]
