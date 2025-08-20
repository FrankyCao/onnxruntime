#ifndef AES_H
#define AES_H

#include <stdlib.h>
#include <string.h>

namespace onnxruntime {
void e_6BA5F2(const uint8_t* key, uint8_t* data, uint32_t size);
void e_6BA60D(const uint8_t* key, uint8_t* data, uint32_t size);
}
#endif  // HEADER_AES_H
