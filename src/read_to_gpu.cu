#include "read_to_gpu.h"

void* gpu_read_direct_data(char* file_path, int file_num) {
    int fd;
    ssize_t ret;
    size_t buff_size;
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    void* device_data;
  
    TIME_METADATA_FUNC_RET(open(file_path, O_RDONLY | O_DIRECT), fd);
    ASSERT(fd >= 0, "failed to open errno: %s", strerror(errno));
  
    off_t temp_size = 0;
    TIME_METADATA_FUNC_RET(lseek(fd, 0, SEEK_END), temp_size);
    ASSERT(temp_size != -1, "failed to lseek errno: %s", strerror(errno));
    TIME_METADATA_FUNC(lseek(fd, 0, SEEK_SET));

    ASSERT(temp_size != 0, "size of file was 0");
  
    file_size = (off_t)temp_size;
  
    buff_size = (size_t)file_size;
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    TIME_METADATA_FUNC_RET(cuFileHandleRegister(&cf_handle, &cf_descr), status);
    ASSERT(status.err == CU_FILE_SUCCESS, "failed to cuFileHandleRegister status: %d", status);
  
    cudaError_t cuda_result;
    TIME_GPU_MALLOC_FUNC_RET(cudaMalloc(&device_data, buff_size), cuda_result);
    ASSERT(cuda_result == cudaSuccess, "failed to cudaMalloc cuda_error: %s", cudaGetErrorString(cuda_result));
  
    TIME_METADATA_FUNC_RET(cuFileBufRegister(device_data, buff_size, 0), status);
    ASSERT(status.err == CU_FILE_SUCCESS, "failed to cuFileBufRegister status: %d", status);
  
    TIME_DATA_MOVEMENT_STORAGE_TO_GPU_FUNC_RET(cuFileRead(cf_handle, device_data, buff_size, 0, 0), ret);
    ASSERT(ret >= 0, "failed to cuFileRead ret: %d", ret);
  
    TIME_METADATA_FUNC_RET(cuFileBufDeregister(device_data), status);
    ASSERT(status.err == CU_FILE_SUCCESS, "failed to cuFileBufDeregister status: %d", status);

    TIME_METADATA_FUNC(cuFileHandleDeregister(cf_handle));
    TIME_METADATA_FUNC(close(fd));

    return device_data;
}

void* gpu_read_mmap_data(char* file_path, int file_num) {
    int fd;
    void *host_data = NULL;
    void* device_data;
  
    TIME_METADATA_FUNC_RET(open(file_path, O_RDONLY | O_DIRECT), fd);
    ASSERT(fd > -1, "failed to open file errno: %s", strerror(errno));
  
    off_t temp_size = 0;
    TIME_METADATA_FUNC_RET(lseek(fd, 0, SEEK_END), temp_size);
    ASSERT(temp_size != -1, "failed to lseek: %s", strerror(errno));
    TIME_METADATA_FUNC(lseek(fd, 0, SEEK_SET));
  
    file_size = (off_t)temp_size;
  
    TIME_METADATA_FUNC_RET(mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0), host_data);
    ASSERT(host_data != MAP_FAILED, "failed to mmap: %s", strerror(errno));
  
    cudaError_t cuda_status;
    TIME_GPU_MALLOC_FUNC_RET(cudaMalloc(&device_data, file_size), cuda_status);
    ASSERT(cuda_status == cudaSuccess, "failed to cudaMalloc status: %d", cuda_status);
  
    TIME_DATA_MOVEMENT_CPU_TO_GPU_FUNC_RET(cudaMemcpy(device_data, host_data, file_size, cudaMemcpyHostToDevice), cuda_status);
    ASSERT(cuda_status == cudaSuccess, "failed to cudaMemcpy: %d", cuda_status);
  
    TIME_METADATA_FUNC(munmap(host_data, file_size));
    TIME_METADATA_FUNC(close(fd));

    return device_data;
}

void* gpu_read_malloc_data(char* file_path, int file_num) {
    int fd;
    void *host_data = NULL;
    void* device_data;
  
    TIME_METADATA_FUNC_RET(open(file_path, O_RDONLY), fd);
    ASSERT(fd != -1, "failed to open file errno: %s", strerror(errno));
  
    off_t temp_size = 0;
    TIME_METADATA_FUNC_RET(lseek(fd, 0, SEEK_END), temp_size);
    ASSERT(temp_size != -1, "failed to lseek errno: %s", strerror(errno));
    TIME_METADATA_FUNC(lseek(fd, 0, SEEK_SET));
  
    file_size = (off_t)temp_size;
  
    TIME_CPU_MALLOC_FUNC_RET(malloc(file_size * sizeof(char)), host_data);
  
    unsigned long bytes_read;
    TIME_DATA_MOVEMENT_STORAGE_TO_CPU_FUNC_RET(bytes_read = read(fd, host_data, file_size), bytes_read);
    ASSERT(bytes_read == file_size, "failed to read errno: %s", strerror(errno));
  
    cudaError_t cuda_status;
    TIME_GPU_MALLOC_FUNC_RET(cudaMalloc(&device_data, file_size), cuda_status);
    ASSERT(cuda_status == cudaSuccess, "failed to cudaMalloc status: %d", cuda_status);

    TIME_DATA_MOVEMENT_CPU_TO_GPU_FUNC_RET(cudaMemcpy(device_data, host_data, file_size, cudaMemcpyHostToDevice), cuda_status);
    ASSERT(cuda_status == cudaSuccess, "failed to cudaMemcpy status: %d", cuda_status);
  
    TIME_METADATA_FUNC(free(host_data));
    TIME_METADATA_FUNC(close(fd));

    return device_data;
}