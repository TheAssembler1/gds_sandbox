#include "read_to_gpu.h"

void* gpu_read_direct_data(char* filepath, int file_num, size_t* file_size) {
    int fd;
    ssize_t ret;
    size_t buff_size;
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    void* device_data;
  
    fd = open(filepath, O_RDONLY | O_DIRECT);
    ASSERT(fd > 0, "failed to open errno: %s", strerror(errno));
  
    off_t temp_size = lseek(fd, 0, SEEK_END);
    ASSERT(temp_size != -1, "failed to lseek errno: %s", strerror(errno));
    lseek(fd, 0, SEEK_SET);
  
    *file_size = (off_t)temp_size;
  
    buff_size = (size_t)*file_size;
    memset(&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    ASSERT(status.err == CU_FILE_SUCCESS, "failed to cuFileHandleRegister status: %d", status);
  
    cudaError_t cuda_result = cudaMalloc(&device_data, buff_size);
    ASSERT(cuda_result == cudaSuccess, "failed to cudaMalloc cuda_error: %s", cudaGetErrorString(cuda_result));
  
    status = cuFileBufRegister(device_data, buff_size, 0);
    ASSERT(status.err == CU_FILE_SUCCESS, "failed to cuFileBufRegister status: %d", status);
  
    ret = cuFileRead(cf_handle, device_data, buff_size, 0, 0);
    ASSERT(ret >= 0, "failed to cuFileRead ret: %d", ret);
  
    status = cuFileBufDeregister(device_data);
    ASSERT(status.err == CU_FILE_SUCCESS, "failed to cuFileBufDeregister status: %d", status);

    cuFileHandleDeregister(cf_handle);
    close(fd);

    return device_data;
}

void* gpu_read_mmap_data(char* file_path, int file_num, size_t* file_size) {
    int fd;
    void *host_data = NULL;
    void* device_data;
  
    fd = open(file_path, O_RDONLY);
    ASSERT(fd > -1, "failed to open file errno: %s", strerror(errno));
  
    off_t temp_size = lseek(fd, 0, SEEK_END);
    ASSERT(temp_size != -1, "failed to lseek: %s", strerror(errno));
    TIME_METADATA_FUNC(lseek(fd, 0, SEEK_SET));
  
    *file_size = (off_t)temp_size;
  
    host_data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ASSERT(host_data != MAP_FAILED, "failed to mmap: %s", strerror(errno));
  
    cudaError_t cuda_status = cudaMalloc(&device_data, *file_size);
    ASSERT(cuda_status == cudaSuccess, "failed to cudaMalloc status: %d", cuda_status);
  
    cuda_status = cudaMemcpy(device_data, host_data, *file_size, cudaMemcpyHostToDevice);
    ASSERT(cuda_status == cudaSuccess, "failed to cudaMemcpy: %d", cuda_status);
  
    munmap(host_data, *file_size);
    close(fd);

    return device_data;
}

void* gpu_read_malloc_data(char* file_path, int file_num, size_t* file_size) {
    int fd;
    void *host_data = NULL;
    void* device_data;
  
    fd = open(file_path, O_RDONLY);
    ASSERT(fd != -1, "failed to open file errno: %s", strerror(errno));
  
    off_t temp_size = lseek(fd, 0, SEEK_END);
    ASSERT(temp_size != -1, "failed to lseek errno: %s", strerror(errno));
    TIME_METADATA_FUNC(lseek(fd, 0, SEEK_SET));
  
    *file_size = (off_t)temp_size;
  
    host_data = malloc(*file_size * sizeof(char));
  
    unsigned long bytes_read = read(fd, host_data, *file_size);
    ASSERT(bytes_read == *file_size, "failed to read errno: %s", strerror(errno));
  
    cudaError_t cuda_status = cudaMalloc(&device_data, *file_size);
    ASSERT(cuda_status == cudaSuccess, "failed to cudaMalloc status: %d", cuda_status);

    cuda_status = cudaMemcpy(device_data, host_data, *file_size, cudaMemcpyHostToDevice);
    ASSERT(cuda_status == cudaSuccess, "failed to cudaMemcpy status: %d", cuda_status);
  
    free(host_data);
    close(fd);

    return device_data;
}