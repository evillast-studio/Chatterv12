import { closeFd, getContentFd } from '@vali98/react-native-fs'
import {
    CompletionParams,
    ContextParams,
    initLlama,
    LlamaContext,
    RNLLAMA_MTMD_DEFAULT_MEDIA_MARKER,
} from 'cui-llama.rn'
import { t } from 'i18next'
import { Platform } from 'react-native'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

import { ModelDataType } from '@db/schema'
import { Storage } from '@lib/enums/Storage'
import { AppDirectory, fileExists, readableFileSize, writeBase64File } from '@lib/utils/File'

import { checkGGMLDeprecated } from './GGML'
import { KV, Model } from './Model'
import { AppSettings } from '../../constants/GlobalValues'
import { Logger } from '../../state/Logger'
import { createMMKVStorage, mmkv } from '../../storage/MMKV'

export type CompletionTimings = {
    predicted_per_token_ms: number
    predicted_per_second: number | null
    predicted_ms: number
    predicted_n: number
    prompt_per_token_ms: number
    prompt_per_second: number | null
    prompt_ms: number
    prompt_n: number
}

export type CompletionOutput = {
    text: string
    timings: CompletionTimings
}

export type LlamaState = {
    context: LlamaContext | undefined
    model?: ModelDataType
    mmproj?: ModelDataType
    loadProgress: number
    chatCount: number
    promptCache?: string
    load: (model: ModelDataType) => Promise<void>
    loadMmproj: (model: ModelDataType) => Promise<void>
    setLoadProgress: (progress: number) => void
    unload: () => Promise<void>
    unloadMmproj: () => Promise<void>
    saveKV: (prompt: string | undefined, media_paths?: string[]) => Promise<void>
    loadKV: () => Promise<boolean>
    completion: (
        params: CompletionParams,
        callback: (text: string) => void,
        completed: (text: string, timings: CompletionTimings) => void
    ) => Promise<void>
    stopCompletion: () => Promise<void>
    tokenLength: (text: string, mediaPaths?: string[]) => Promise<number>
    tokenize: (text: string, media_paths?: string[]) => Promise<{ tokens: number[] } | undefined>
}

export type LlamaConfig = {
    context_length: number
    threads: number
    gpu_layers: number
    batch: number
    ubatch: number
    ctx_shift: boolean
    devices: string[]
    use_mmap: boolean
    use_mlock: boolean
    flash_attn: boolean
    kv_unified: boolean
    cache_type_k: string
    cache_type_v: string
    no_extra_bufts: boolean
}

export type EngineDataProps = {
    config: LlamaConfig
    lastModel?: ModelDataType
    lastMmproj?: ModelDataType
    setConfiguration: (config: LlamaConfig) => void
    setLastModelLoaded: (model: ModelDataType | undefined) => void
    setLastMmprojLoaded: (model: ModelDataType | undefined) => void
    maybeClearLastLoaded: (mode: ModelDataType) => void
}

const sessionFile = `${AppDirectory.SessionPath}llama-session.bin`

const defaultConfig: LlamaConfig = {
    context_length: 2048,
    threads: 4,
    gpu_layers: 0,
    batch: 512,
    ubatch: 512,
    ctx_shift: true,
    use_mmap: false,
    use_mlock: false,
    flash_attn: false,
    kv_unified: true,
    cache_type_k: 'q8_0',
    cache_type_v: 'q8_0',
    no_extra_bufts: false,
    devices: [],
}

export namespace Llama {
    export const useLlamaPreferencesStore = create<EngineDataProps>()(
        persist(
            (set, get) => ({
                config: defaultConfig,
                setConfiguration: (config: LlamaConfig) => {
                    set({ config: config })
                },
                setLastModelLoaded: (model: ModelDataType | undefined) => {
                    if (get().lastModel?.id === model?.id) return
                    set({ lastModel: model, lastMmproj: undefined })
                },
                setLastMmprojLoaded: (mmproj: ModelDataType | undefined) => {
                    set({ lastMmproj: mmproj })
                },
                maybeClearLastLoaded: (data) => {
                    if (data.id === get().lastModel?.id) {
                        set({ lastModel: undefined, lastMmproj: undefined })
                    } else if (data.id === get().lastMmproj?.id) {
                        set({ lastMmproj: undefined })
                    }
                },
            }),
            {
                name: Storage.EngineData,
                partialize: (state) => ({
                    config: state.config,
                    lastModel: state.lastModel,
                    lastMmproj: state.lastMmproj,
                }),
                storage: createMMKVStorage(),
                version: 4,
                migrate: (persistedState: any, version) => {
                    if (version === 1) {
                        persistedState.config.ctx_shift = true
                        Logger.info('Migrated EngineData to v2')
                    }
                    if (version === 2) {
                        persistedState.config.devices = []
                        Logger.info('Migrated EngineData to v3')
                    }
                    if (version === 3) {
                        const c = persistedState.config
                        if (c.flash_attn === undefined) c.flash_attn = false
                        if (c.ubatch === undefined) c.ubatch = c.batch ?? 512
                        if (c.use_mmap === undefined) c.use_mmap = Platform.OS === 'android' ? false : true
                        if (c.use_mlock === undefined) c.use_mlock = false
                        if (c.kv_unified === undefined) c.kv_unified = true
                        if (c.no_extra_bufts === undefined) c.no_extra_bufts = false
                        if (c.cache_type_k === undefined) c.cache_type_k = 'q8_0'
                        if (c.cache_type_v === undefined) c.cache_type_v = 'q8_0'
                        Logger.info('Migrated EngineData to v4 (PocketPal params)')
                    }
                    return persistedState
                },
            }
        )
    )

    export const useLlamaModelStore = create<LlamaState>()((set, get) => ({
        context: undefined,
        loadProgress: 0,
        chatCount: 0,
        promptCache: undefined,
        load: async (model: ModelDataType) => {
            const config = useLlamaPreferencesStore.getState().config

            if (get()?.model?.id === model.id) {
                return Logger.errorToast(t('model.toast.modelAlreadyLoaded'))
            }

            if (checkGGMLDeprecated(parseInt(model.quantization))) {
                return Logger.errorToast(t('model.toast.quantizationNoLongerSupported'))
            }

            if (!(await Model.getModelExists(model.file_path))) {
                Logger.errorToast(t('model.toast.modelDoesNotExist'))
                Model.verifyModelList()
                return
            }

            if (get().context !== undefined) {
                await get().unload()
            }

            let model_path = model.file_path
            if (model.file_path.includes('content://')) {
                model_path = (await getContentFd(model_path)) ?? model_path
            }

            const params: ContextParams = {
                model: model_path,
                n_ctx: config.context_length,
                n_threads: config.threads,
                n_batch: config.batch,
                n_ubatch: config.ubatch,
                ctx_shift: config.ctx_shift,
                n_gpu_layers: config.gpu_layers,
                use_mlock: config.use_mlock,
                use_mmap: config.use_mmap,
                devices: config.devices,
                flash_attn: config.flash_attn,
                // Cache KV cuantizado: funciona con flash_attn OFF
                cache_type_k: config.cache_type_k as any,
                cache_type_v: config.cache_type_v as any,
                // PocketPal v2.0: une buferes KV, ahorra mucha RAM
                kv_unified: config.kv_unified as any,
                // PocketPal v2.2: false = repack de pesos ON
                no_extra_bufts: config.no_extra_bufts as any,
            }

            Logger.info(
                `\n------ MODEL LOAD -----\n` +
                `Model: ${model.name}\n` +
                `Context: ${params.n_ctx} | Threads: ${params.n_threads}\n` +
                `Batch: ${params.n_batch} | uBatch: ${params.n_ubatch}\n` +
                `GPU Layers: ${params.n_gpu_layers}\n` +
                `use_mmap: ${params.use_mmap} | use_mlock: ${params.use_mlock}\n` +
                `flash_attn: ${params.flash_attn} | kv_unified: ${params.kv_unified}\n` +
                `cache_type_k: ${params.cache_type_k} | cache_type_v: ${params.cache_type_v}\n` +
                `no_extra_bufts: ${params.no_extra_bufts}`
            )

            const progressCallback = (progress: number) => {
                if (progress % 5 === 0) get().setLoadProgress(progress)
            }

            const llamaContext = await initLlama(params, progressCallback).catch((error) => {
                Logger.errorToast(t('model.toast.couldNotLoadModel'), JSON.stringify(error))
                if (model.file_path.includes('content://')) {
                    closeFd(model_path)
                }
            })

            if (!llamaContext) return

            set({
                context: llamaContext,
                model: model,
                chatCount: 1,
            })

            useLlamaPreferencesStore.getState().setLastModelLoaded(model)
            KV.useKVStore.getState().setKvCacheLoaded(false)
        },
        loadMmproj: async (model: ModelDataType) => {
            const context = get().context
            if (!context) return

            let model_path = model.file_path
            if (model.file_path.includes('content://')) {
                model_path = (await getContentFd(model_path)) ?? model_path
            }

            Logger.info('Loading MMPROJ')
            await context.initMultimodal({ path: model_path, use_gpu: true }).catch((e) => {
                if (model.file_path.includes('content://')) {
                    closeFd(model_path)
                }
                Logger.errorToast(t('model.toast.failedToLoadMMPROJ'), JSON.stringify(e))
            })
            if (await context.isMultimodalEnabled()) {
                const capabilities = await context.getMultimodalSupport()
                Logger.info(
                    `MMPROJ Loaded:\n- Vision: ${capabilities.vision}\n- Audio: ${capabilities.audio}`
                )
            }

            set({ mmproj: model })
            useLlamaPreferencesStore.getState().setLastMmprojLoaded(model)
        },
        setLoadProgress: (progress: number) => {
            set({ loadProgress: progress })
        },
        unload: async () => {
            if (get().mmproj) {
                await get().context?.releaseMultimodal()
            }
            await get().context?.release()
            set({ context: undefined, model: undefined, mmproj: undefined })
            Logger.info('Model Unloaded')
        },
        unloadMmproj: async () => {
            if (!get().mmproj) return
            await get()
                .context?.releaseMultimodal()
                .catch((e) => {
                    Logger.errorToast(t('model.toast.failedToUnloadMMPROJ'), JSON.stringify(e))
                })
            set({ mmproj: undefined })
        },
        completion: async (
            params: CompletionParams,
            callback = (text: string) => {},
            completed = (text: string) => {}
        ) => {
            const llamaContext = get().context
            if (llamaContext === undefined) {
                Logger.errorToast(t('model.toast.noModelLoaded'))
                return
            }

            return llamaContext
                .completion(params, (data) => {
                    callback(data.token)
                })
                .then(async ({ text, timings }: CompletionOutput) => {
                    completed(text, timings)
                    Logger.info(
                        `\n---- Start Chat ${get().chatCount} ----\n${textTimings(timings)}\n---- End Chat ${get().chatCount} ----\n`
                    )
                    set({ chatCount: get().chatCount + 1 })
                    if (mmkv.getBoolean(AppSettings.SaveLocalKV)) {
                        await get().saveKV(params.prompt, params.media_paths ?? [])
                    }
                })
        },
        stopCompletion: async () => {
            await get().context?.stopCompletion()
        },
        saveKV: async (prompt, media_paths) => {
            const llamaContext = get().context
            if (!llamaContext) {
                Logger.errorToast(t('model.toast.noModelLoaded'))
                return
            }

            if (prompt) {
                const tokens = (await get().tokenize(prompt, media_paths ?? []))?.tokens
                KV.useKVStore.getState().setKvCacheTokens(tokens ?? [])
            }

            if (!fileExists(sessionFile)) {
                Logger.warn('Session file does not exist, creating...')
                await writeBase64File(sessionFile, '')
            }

            const now = performance.now()
            const data = await llamaContext.saveSession(sessionFile.replace('file://', ''))
            Logger.info(
                data === -1
                    ? 'Failed to save KV cache'
                    : `Saved KV in ${Math.floor(performance.now() - now)}ms with ${data} tokens`
            )
            Logger.info(`Current KV Size is: ${readableFileSize(await KV.getKVSize())}`)
        },
        loadKV: async () => {
            let result = false
            const llamaContext = get().context
            if (!llamaContext) {
                Logger.errorToast(t('model.toast.noModelLoaded'))
                return false
            }
            if (!fileExists(sessionFile)) {
                Logger.warn('No Cache found')
                return false
            }
            await llamaContext
                .loadSession(sessionFile.replace('file://', ''))
                .then(() => {
                    Logger.info('Session loaded from KV cache')
                    result = true
                })
                .catch(() => {
                    Logger.error('Session loaded could not load from KV cache')
                })
            return result
        },
        tokenLength: async (text: string, mediaPaths: string[] = []) => {
            const finalPaths = get().mmproj ? mediaPaths : []
            if (!get().mmproj && mediaPaths.length > 0) {
                Logger.warnToast(t('model.toast.mediaAddedWithoutMMPROJModel'))
            }
            const result = await get().context?.tokenize(
                text + finalPaths.map(() => RNLLAMA_MTMD_DEFAULT_MEDIA_MARKER).join(),
                {
                    media_paths: finalPaths.map((item) => item.replace('file://', '')),
                }
            )
            if (!result) return 0
            return result.tokens.length
        },
        tokenize: async (text: string, media_paths: string[] = []) => {
            const params = get().mmproj ? { media_paths } : {}
            return await get().context?.tokenize(text, params)
        },
    }))

    const textTimings = (timings: CompletionTimings) => {
        return (
            `\n[Prompt Timings]` +
            (timings.prompt_n > 0
                ? `\nPrompt Per Token: ${timings.prompt_per_token_ms.toFixed(2)} ms/token` +
                  `\nPrompt Per Second: ${timings.prompt_per_second?.toFixed(2) ?? 0} tokens/s` +
                  `\nPrompt Time: ${(timings.prompt_ms / 1000).toFixed(2)}s` +
                  `\nPrompt Tokens: ${timings.prompt_n} tokens`
                : '\nNo Tokens Processed') +
            `\n\n[Predicted Timings]` +
            (timings.predicted_n > 0
                ? `\nPredicted Per Token: ${timings.predicted_per_token_ms.toFixed(2)} ms/token` +
                  `\nPredicted Per Second: ${timings.predicted_per_second?.toFixed(2) ?? 0} tokens/s` +
                  `\nPrediction Time: ${(timings.predicted_ms / 1000).toFixed(2)}s` +
                  `\nPredicted Tokens: ${timings.predicted_n} tokens\n`
                : '\nNo Tokens Generated')
        )
    }
    }
