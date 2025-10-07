"use client"

import { useMemo, useState } from 'react'
import {
  Box,
  Container,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Switch,
  Slider,
  Button,
  Stack,
  Chip,
  Divider,
} from '@mui/material'
import { useSiteConfig } from '@/lib/hooks/useSiteConfig'
import { useFeatureFlags } from '@/contexts/FeatureFlagContext'
import { getDisabledModules } from '@/lib/security/disabled-modules'

const sliderMarks = [
  { value: 0, label: '0%' },
  { value: 25, label: '25%' },
  { value: 50, label: '50%' },
  { value: 75, label: '75%' },
  { value: 100, label: '100%' },
]

export default function SiteConfigAdminPage() {
  const { config, loading } = useSiteConfig()
  const { flags, setOverride, setRollout, scheduleRollback, clearOverride } = useFeatureFlags()
  const [selectedFlag, setSelectedFlag] = useState<string | null>(null)

  const modules = useMemo(() => {
    if (!config?.modules) return []
    return Object.entries(config.modules)
  }, [config])

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 6 }}>
        <Typography variant="body1">Loading site configuration…</Typography>
      </Container>
    )
  }

  return (
    <Container maxWidth="lg" sx={{ py: 6 }}>
      <Stack spacing={4}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Site Configuration Control Panel
          </Typography>
          <Typography variant="body2" color="text.secondary">
            View runtime feature flags, module visibility, and rollout levers. Changes here apply to the
            current session via client-side overrides to simulate rollout and rollback procedures.
          </Typography>
        </Box>

        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Feature Flags
          </Typography>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Flag</TableCell>
                <TableCell>Enabled</TableCell>
                <TableCell>Rollout</TableCell>
                <TableCell>Source</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(flags).map(([flag, state]) => (
                <TableRow
                  key={flag}
                  hover
                  selected={selectedFlag === flag}
                  onClick={() => setSelectedFlag(flag)}
                  sx={{ cursor: 'pointer' }}
                >
                  <TableCell sx={{ textTransform: 'snakecase' }}>{flag}</TableCell>
                  <TableCell>
                    <Switch
                      checked={state.enabled}
                      onChange={(_, checked) => setOverride(flag, { enabled: checked })}
                      inputProps={{ 'aria-label': `Toggle ${flag}` }}
                    />
                  </TableCell>
                  <TableCell sx={{ minWidth: 180 }}>
                    <Slider
                      size="small"
                      value={state.rolloutPercentage}
                      step={5}
                      marks={sliderMarks}
                      valueLabelDisplay="auto"
                      onChange={(_, value) =>
                        setRollout(flag, Array.isArray(value) ? value[0] : value)
                      }
                    />
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={state.source === 'override' ? 'override' : 'config'}
                      size="small"
                      color={state.source === 'override' ? 'secondary' : 'default'}
                    />
                  </TableCell>
                  <TableCell align="right">
                    <Stack direction="row" spacing={1} justifyContent="flex-end">
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => scheduleRollback(flag)}
                      >
                        Rollback
                      </Button>
                      <Button size="small" onClick={() => clearOverride(flag)}>
                        Clear
                      </Button>
                    </Stack>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Paper>

        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Module Visibility
          </Typography>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Module</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {modules.map(([module, enabled]) => (
                <TableRow key={module}>
                  <TableCell>{module}</TableCell>
                  <TableCell>
                    <Chip
                      label={enabled ? 'enabled' : 'disabled'}
                      color={enabled ? 'success' : 'default'}
                      size="small"
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <Divider sx={{ my: 2 }} />
          <Typography variant="caption" color="text.secondary">
            Build-time disabled modules: {getDisabledModules().join(', ') || 'none'}
          </Typography>
        </Paper>
      </Stack>
    </Container>
  )
}
