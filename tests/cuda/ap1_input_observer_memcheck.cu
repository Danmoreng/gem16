// Offline memory check of the exact AP1 observer kernels and slot arithmetic.
// Uses captured operator bytes; does not execute or qualify model kernels.
#include "ap1_input_observer.cu"
#include <algorithm>
#include <iostream>

int main(int argc, char** argv) {
  try {
    if (argc < 2 || enabled) throw std::runtime_error("pass capture directories; do not enable interposition");
    Check(cudaMalloc(&arena, kBytes));
    Check(cudaMemset(arena, 0, kBytes));
    std::byte* source = nullptr;
    Check(cudaMalloc(&source, kGlobalStride + 512U));
    std::vector<std::byte> input(kGlobalStride + 512U), output(input.size());
    std::size_t files = 0, total = 0;
    auto exercise = [&](std::size_t bytes) {
      const unsigned slot = bytes <= kSmallStride - 512U ? 3U : 0U;
      const auto capacity = slot < 3 ? kGlobalStride : kSmallStride;
      Check(cudaMemset(Buffer(0, slot), 0x5a, capacity));
      Check(cudaMemcpy(source, input.data(), bytes + 512U, cudaMemcpyHostToDevice));
      Record record{};
      // A nonzero first part also checks Add's 256-byte alignment arithmetic.
      Add(slot, record, "prefix", source, 1U, nullptr);
      Add(slot, record, "actual-input", source + 256U, bytes, nullptr);
      if (record.count != 2U || record.parts[1].offset != 256U || record.used != bytes + 256U)
        throw std::runtime_error("snapshot extent/alignment mismatch");
      Ap1MarkSnapshotKernel<<<1U, 1U>>>(Marker(0, slot));
      Check(cudaGetLastError());
      Copy(Buffer(0, slot), Buffer(1, slot), record.used, nullptr);
      Copy(Marker(0, slot), Marker(1, slot), sizeof(unsigned), nullptr);
      Check(cudaDeviceSynchronize());
      Check(cudaMemcpy(output.data(), Buffer(1, slot), record.used, cudaMemcpyDeviceToHost));
      if (output[0] != input[0] || !std::equal(input.begin() + 256U, input.begin() + 256U + bytes,
                                             output.begin() + 256U))
        throw std::runtime_error("snapshot byte mismatch");
      for (std::size_t i = 1; i < 256U; ++i)
        if (output[i] != std::byte{0x5a}) throw std::runtime_error("alignment padding overwritten");
      unsigned marker = 0;
      Check(cudaMemcpy(&marker, Marker(1, slot), sizeof(marker), cudaMemcpyDeviceToHost));
      if (marker != 1U) throw std::runtime_error("marker mismatch");
      std::byte sentinel{};
      Check(cudaMemcpy(&sentinel, Buffer(0, slot) + record.used, 1U, cudaMemcpyDeviceToHost));
      if (sentinel != std::byte{0x5a}) throw std::runtime_error("snapshot overran extent");
    };
    for (std::size_t n : {1U, 2U, 31U, 255U, 256U, 257U, 4097U}) {
      for (std::size_t i = 0; i < n + 512U; ++i) input[i] = std::byte(i % 251U);
      exercise(n);
    }
    for (int arg = 1; arg < argc; ++arg) {
      for (const auto& item : std::filesystem::directory_iterator(argv[arg])) {
        if (item.path().extension() != ".bin") continue;
        if (item.is_symlink() || !item.is_regular_file()) throw std::runtime_error("invalid capture file");
        const auto bytes = item.file_size();
        if (bytes == 0 || bytes > kGlobalStride - 512U || ++files > 1024U)
          throw std::runtime_error("capture size/count exceeds test bound");
        std::fill(input.begin(), input.end(), std::byte{0xa5});
        std::ifstream stream(item.path(), std::ios::binary);
        stream.read(reinterpret_cast<char*>(input.data() + 256U), bytes);
        if (!stream || stream.peek() != std::char_traits<char>::eof())
          throw std::runtime_error("capture read/size mismatch");
        exercise(bytes);
        total += bytes;
      }
    }
    if (files == 0) throw std::runtime_error("no captured bytes checked");
    // Exercise markers at every actual reserved slot/set boundary as well.
    for (unsigned set = 0; set < 4U; ++set)
      for (unsigned slot = 0; slot < 7U; ++slot) {
        Ap1MarkSnapshotKernel<<<1U, 1U>>>(Marker(set, slot));
        Check(cudaGetLastError());
      }
    Check(cudaDeviceSynchronize());
    Check(cudaFree(source)); Check(cudaFree(arena)); arena = nullptr;
    std::cout << "{\"passed\":true,\"captured_files\":" << files
              << ",\"captured_bytes\":" << total
              << ",\"boundary_lengths\":[1,2,31,255,256,257,4097],\"marker_slots\":28}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
