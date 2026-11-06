import socket

_original_getaddrinfo = socket.getaddrinfo

def ipv4_getaddrinfo(*args, **kwargs):
    results = _original_getaddrinfo(*args, **kwargs)

    ipv4_results = [
        result for result in results
        if result[0] == socket.AF_INET
    ]

    return ipv4_results or results

socket.getaddrinfo = ipv4_getaddrinfo