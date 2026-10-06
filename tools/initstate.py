#!/usr/bin/env python3
# Estado inicial da thread raiz (module_start) para executar sem trace de referência.
# Reproduz o que o kernel do PSP faz ao iniciar o módulo principal (semântica de
# __KernelSetupRootThread / PSPThread::FillStack no PPSSPP):
#   - pilha do tamanho declarado em module_start_thread_parameter (padrão 0x40000),
#     preenchida com 0xFF salvo PSP_THREAD_ATTR_NO_FILLSTACK; os 256 bytes do topo são o
#     bloco k0 da thread (uid em +0xc0, início da pilha em +0xc8, -1 em +0xf8/+0xfc);
#   - a0 = tamanho dos argumentos, a1 = cópia deles na pilha (caminho do EBOOT, com NUL);
#   - sp desce mais 64 bytes; gp = gp_value do SceModuleInfo.
# O ra não é semeado: no código recompilado o retorno de module_start é um return nativo.
#
# Uso: initstate.py <prx-elf> <base-hex> <out.trace> [--stack-top=HEX] [--uid=HEX]
import struct
import sys
from analyze import Elf

NID_MODULE_START_THREAD_PARAM = 0x0F7C276C
PSP_THREAD_ATTR_NO_FILLSTACK = 0x00100000
BOOT_PATH = b"disc0:/PSP_GAME/SYSDIR/EBOOT.BIN"


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    opts = dict(o[2:].split("=", 1) for o in argv[1:] if o.startswith("--") and "=" in o)
    if len(args) < 3:
        sys.stderr.write("uso: initstate.py <prx-elf> <base-hex> <out.trace> [--stack-top=HEX] [--uid=HEX]\n")
        return 2
    elf = Elf(args[0], base=int(args[1], 16))
    stack_top = int(opts.get("stack-top", "0a000000"), 16)   # topo da RAM de usuário
    uid = int(opts.get("uid", "110"), 16)                     # primeiro UID atribuído por sched.c

    def r32(a):
        return struct.unpack("<I", elf.read_at_vaddr(a, 4))[0]

    mi = elf.sec(".rodata.sceModuleInfo")
    if mi is None:
        sys.stderr.write("SceModuleInfo não encontrado\n")
        return 1
    gp, libent, libentend = r32(mi["addr"] + 32), r32(mi["addr"] + 36), r32(mi["addr"] + 40)

    prio, stacksize, attr = 0x20, 0x40000, 0
    pos = libent
    while pos < libentend:
        name, ver, eattr, size, nvar, nfunc, tbl = struct.unpack("<IHHBBHI", elf.read_at_vaddr(pos, 16))
        if size == 0:
            break
        n = nvar + nfunc
        for i in range(n):
            if r32(tbl + 4 * i) == NID_MODULE_START_THREAD_PARAM:
                p = r32(tbl + 4 * (n + i))
                prio = r32(p + 4) or prio
                stacksize = r32(p + 8) or stacksize
                attr = r32(p + 12)
        pos += size * 4

    mem = {}
    start = stack_top - stacksize
    if not attr & PSP_THREAD_ATTR_NO_FILLSTACK:
        for a in range(start, stack_top, 4):
            mem[a] = 0xFFFFFFFF
    sp = stack_top - 256
    k0 = sp
    for a in range(k0, k0 + 0x100, 4):
        mem[a] = 0
    mem[k0 + 0xC0] = uid
    mem[k0 + 0xC8] = start
    mem[k0 + 0xF8] = mem[k0 + 0xFC] = 0xFFFFFFFF
    mem[start] = uid

    argb = BOOT_PATH + b"\0"
    a0 = len(argb)
    sp -= (a0 + 0xF) & ~0xF
    a1 = sp
    argb += b"\0" * (-len(argb) % 4)
    for i in range(0, len(argb), 4):
        mem[a1 + i] = struct.unpack("<I", argb[i:i + 4])[0]
    sp -= 64

    toks = [f"r4=0x{a0:08x}", f"r5=0x{a1:08x}", f"r26=0x{k0:08x}", f"r28=0x{gp:08x}", f"r29=0x{sp:08x}"]
    toks += [f"m32[0x{a:08x}]=0x{v:08x}" for a, v in sorted(mem.items())]
    with open(args[2], "w", encoding="ascii", newline="\n") as f:
        f.write("# init " + " ".join(toks) + "\n")
    print(f"thread raiz: prio=0x{prio:x} pilha=0x{stacksize:x} [0x{start:08x}..0x{stack_top:08x}) "
          f"sp=0x{sp:08x} k0=0x{k0:08x} gp=0x{gp:08x} a0={a0} a1=0x{a1:08x}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
