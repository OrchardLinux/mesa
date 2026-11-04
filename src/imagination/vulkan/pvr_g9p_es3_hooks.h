/* SPDX-License-Identifier: MIT */
/* Same dispatch convention as PVR_G5P_HOOK in PR1. */
#ifndef PVR_G9P_ES3_HOOKS_H
#define PVR_G9P_ES3_HOOKS_H
#ifdef PVR_G9P_ES3
#   define PVR_G9P_ES3_HOOK(name, ...) PVR_G9P_ES3_HOOK_##name(__VA_ARGS__)
#else
#   define PVR_G9P_ES3_HOOK(name, ...)
#endif
#endif
