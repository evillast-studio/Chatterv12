import { useFocusEffect } from 'expo-router'
import React, { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { BackHandler, Platform, View } from 'react-native'
import { useMMKVBoolean, useMMKVNumber } from 'react-native-mmkv'
import Animated, { Easing, SlideInRight, SlideOutRight } from 'react-native-reanimated'
import { useShallow } from 'zustand/react/shallow'

import ThemedButton from '@components/buttons/ThemedButton'
import HorizontalSelector from '@components/input/HorizontalSelector'
import ThemedSlider from '@components/input/ThemedSlider'
import ThemedSwitch from '@components/input/ThemedSwitch'
import SectionTitle from '@components/text/SectionTitle'
import Alert from '@components/views/Alert'
import { AppSettings, Global } from '@lib/constants/GlobalValues'
import { Llama } from '@lib/engine/Local/LlamaLocal'
import { KV } from '@lib/engine/Local/Model'
import useBackendDevices from '@lib/hooks/BackendDevices'
import { Logger } from '@lib/state/Logger'
import { readableFileSize } from '@lib/utils/File'

type ModelSettingsProp = {
    modelImporting: boolean
    modelLoading: boolean
    exit: () => void
}

const deviceLabels = { GPUOpenCL: 'OpenCL', HTP0: 'Hexagon', CPU: 'CPU' }

const ModelSettings: React.FC<ModelSettingsProp> = ({ modelImporting, modelLoading, exit }) => {
    const { t } = useTranslation()
    const { config, setConfig } = Llama.useLlamaPreferencesStore(
        useShallow((state) => ({
            config: state.config,
            setConfig: state.setConfiguration,
        }))
    )

    const devices = useBackendDevices()

    const [saveKV, setSaveKV] = useMMKVBoolean(AppSettings.SaveLocalKV)
    const [autoloadLocal, setAutoloadLocal] = useMMKVBoolean(AppSettings.AutoLoadLocal)
    const [showModelInChat, setShowModelInChat] = useMMKVBoolean(AppSettings.ShowModelInChat)
    const [threadCount] = useMMKVNumber(Global.CPUThreads)

    const [kvSize, setKVSize] = useState(0)

    const getKVSize = async () => {
        const size = await KV.getKVSize()
        setKVSize(size)
    }

    useEffect(() => {
        KV.getKVSize().then(setKVSize)
    }, [])

    const backAction = () => {
        exit()
        return true
    }

    useFocusEffect(() => {
        const handler = BackHandler.addEventListener('hardwareBackPress', backAction)
        return () => handler.remove()
    })

    const handleDeleteKV = () => {
        Alert.alert({
            title: t('model.alert.deletekv.title'),
            description: t('model.alert.deletekv.description', { size: readableFileSize(kvSize) }),
            buttons: [
                { label: t('common.actions.delete') },
                {
                    label: t('model.alert.deletekv.title'),
                    onPress: async () => {
                        await KV.deleteKV()
                        Logger.info(t('model.toast.deletekv'))
                        getKVSize()
                    },
                    type: 'warning',
                },
            ],
        })
    }

    return (
        <Animated.ScrollView
            showsVerticalScrollIndicator={false}
            style={{ flex: 1 }}
            entering={SlideInRight.easing(Easing.inOut(Easing.cubic))}
            exiting={SlideOutRight.easing(Easing.inOut(Easing.cubic))}>
            <SectionTitle>{t('model.settings.cpu')}</SectionTitle>
            <View style={{ marginTop: 16 }} />
            {config && (
                <>
                    <ThemedSlider
                        label={t('model.maxcontext')}
                        value={config.context_length}
                        onValueChange={(value) => setConfig({ ...config, context_length: value })}
                        min={1024}
                        max={32768}
                        step={1024}
                        disabled={modelImporting || modelLoading}
                    />
                    <ThemedSlider
                        label={t('model.threads')}
                        value={config.threads}
                        onValueChange={(value) => setConfig({ ...config, threads: value })}
                        min={1}
                        max={threadCount ?? 8}
                        step={1}
                        disabled={modelImporting || modelLoading}
                    />

                    <ThemedSlider
                        label={t('model.batch')}
                        value={config.batch}
                        onValueChange={(value) => setConfig({ ...config, batch: value })}
                        min={16}
                        max={1024}
                        step={16}
                        disabled={modelImporting || modelLoading}
                    />

                    <ThemedSlider
                        label="uBatch (PocketPal)"
                        value={config.ubatch ?? config.batch}
                        onValueChange={(value) => setConfig({ ...config, ubatch: value })}
                        min={16}
                        max={1024}
                        step={16}
                        disabled={modelImporting || modelLoading}
                    />

                    <ThemedSwitch
                        label="use_mmap (PocketPal)"
                        value={config.use_mmap ?? false}
                        onChangeValue={(value) => setConfig({ ...config, use_mmap: value })}
                        description="Desactivar en Android ARM activa repack (más rápido en Kirin)"
                    />

                    <ThemedSwitch
                        label="Memory Lock (PocketPal)"
                        value={config.use_mlock ?? false}
                        onChangeValue={(value) => setConfig({ ...config, use_mlock: value })}
                        description="Fuerza el modelo en RAM, evita swap (recomendado en Kirin)"
                    />

                    <ThemedSwitch
                        label="Flash Attention (PocketPal)"
                        value={config.flash_attn ?? false}
                        onChangeValue={(value) => setConfig({ ...config, flash_attn: value })}
                        description="Desactivar en Kirin 710 (Mali-G51 no lo soporta bien)"
                    />

                    <HorizontalSelector
                        style={{ paddingBottom: 12 }}
                        label="Cache KV tipo K (PocketPal)"
                        values={[
                            { label: 'f16', value: 'f16' },
                            { label: 'q8_0', value: 'q8_0' },
                            { label: 'q4_0', value: 'q4_0' },
                        ]}
                        selected={config.cache_type_k ?? 'q8_0'}
                        onPress={(value) => setConfig({ ...config, cache_type_k: value })}
                    />

                    <HorizontalSelector
                        style={{ paddingBottom: 12 }}
                        label="Cache KV tipo V (PocketPal)"
                        values={[
                            { label: 'f16', value: 'f16' },
                            { label: 'q8_0', value: 'q8_0' },
                            { label: 'q4_0', value: 'q4_0' },
                        ]}
                        selected={config.cache_type_v ?? 'q8_0'}
                        onPress={(value) => setConfig({ ...config, cache_type_v: value })}
                    />

                    {/* Note: llama.rn does not have any Android gpu acceleration */}
                    {(Platform.OS === 'ios' || devices.length > 1) && (
                        <ThemedSlider
                            label={t('model.gpulayers')}
                            value={config.gpu_layers}
                            onValueChange={(value) => setConfig({ ...config, gpu_layers: value })}
                            min={0}
                            max={100}
                            step={1}
                            disabled={modelImporting || modelLoading}
                        />
                    )}

                    <ThemedSwitch
                        label={t('model.contextshift')}
                        value={config.ctx_shift}
                        onChangeValue={(value) => {
                            setConfig({ ...config, ctx_shift: value })
                        }}
                    />

                    {devices.length > 1 && (
                        <HorizontalSelector
                            style={{ paddingBottom: 12 }}
                            label={t('model.backenddev')}
                            values={devices.map((item) => ({
                                label: deviceLabels[item as keyof typeof deviceLabels] ?? item,
                                value: item,
                            }))}
                            selected={config.devices?.[0]}
                            onPress={(value) => {
                                const devices = value === 'CPU' ? [value] : [value, 'CPU']
                                setConfig({ ...config, devices })
                            }}
                        />
                    )}
                </>
            )}
            <SectionTitle>{t('model.settings.advanced')}</SectionTitle>
            <ThemedSwitch
                label={t('model.modelnamechat')}
                value={showModelInChat}
                onChangeValue={setShowModelInChat}
            />
            <ThemedSwitch
                label={t('model.autoload')}
                value={autoloadLocal}
                onChangeValue={setAutoloadLocal}
            />
            <ThemedSwitch
                label={t('model.savekv')}
                value={saveKV}
                onChangeValue={setSaveKV}
                description={saveKV ? '' : t('model.savekvdesc')}
            />
            {saveKV && (
                <ThemedButton
                    buttonStyle={{ marginTop: 8 }}
                    label={t('model.purgekv', { size: readableFileSize(kvSize) })}
                    onPress={handleDeleteKV}
                    variant={kvSize === 0 ? 'disabled' : 'critical'}
                />
            )}
        </Animated.ScrollView>
    )
}

export default ModelSettings
