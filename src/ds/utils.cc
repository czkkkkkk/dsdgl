#include "utils.h"

#include <cstring>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/ip.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <dmlc/logging.h>

namespace dgl {
namespace ds {

int GetAvailablePort() {
  struct sockaddr_in addr;
  addr.sin_port = htons(0);   // 0 means let system pick up an available port.
  addr.sin_family = AF_INET;  // IPV4
  addr.sin_addr.s_addr = htonl(INADDR_ANY);  // set addr to any interface

  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (0 != bind(sock, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
    DLOG(WARNING) << "bind()";
    return 0;
  }
  socklen_t addr_len = sizeof(struct sockaddr_in);
  if (0 != getsockname(sock, (struct sockaddr*)&addr, &addr_len)) {
    DLOG(WARNING) << "getsockname()";
    return 0;
  }

  int ret = ntohs(addr.sin_port);
  close(sock);
  return ret;
}

std::string GetHostName() {
  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);

  struct addrinfo hints = {0};
  hints.ai_family = AF_UNSPEC;
  hints.ai_flags = AI_CANONNAME;

  struct addrinfo* res = 0;
  std::string fqdn;
  if (getaddrinfo(hostname, 0, &hints, &res) == 0) {
    // The hostname was successfully resolved.
    fqdn = std::string(res->ai_canonname);
    freeaddrinfo(res);
  } else {
    // Not resolved, so fall back to hostname returned by OS.
    LOG(FATAL) << " ERROR: No HostName.";
  }

  return fqdn;
}

void getHostName(char* hostname, int maxlen) {
  if (gethostname(hostname, maxlen) != 0) {
    LOG(FATAL) << "Cannot get hostname";
    return;
  }
  int i = 0;
  while ((hostname[i] != '.') && (hostname[i] != '\0') && (i < maxlen - 1)) i++;
  hostname[i] = '\0';
}

uint64_t getHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname) $(readlink /proc/self/ns/uts)
 */
uint64_t getHostHash(void) {
  char uname[1024];
  // Start off with the hostname
  (void)getHostName(uname, sizeof(uname));
  int hlen = strlen(uname);
  int len =
      readlink("/proc/self/ns/uts", uname + hlen, sizeof(uname) - 1 - hlen);
  if (len < 0) len = 0;

  uname[hlen + len] = '\0';

  return getHash(uname);
}

}
}