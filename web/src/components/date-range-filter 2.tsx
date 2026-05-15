"use client";

import { CalendarIcon } from "lucide-react";

import { Field } from "@/components/ui/field";

type DateRangeFilterProps = {
  startDate: string;
  endDate: string;
  onChange: (startDate: string, endDate: string) => void;
};

export function DateRangeFilter({ startDate, endDate, onChange }: DateRangeFilterProps) {
  return (
    <Field className="w-full sm:w-auto">
      <div className="flex h-auto flex-col gap-2 rounded-xl border border-stone-200 bg-white px-3 py-2 text-stone-700 sm:h-10 sm:flex-row sm:items-center sm:py-0">
        <CalendarIcon className="hidden size-4 shrink-0 text-stone-400 sm:block" />
        <input
          type="date"
          value={startDate}
          onChange={(event) => onChange(event.target.value, endDate)}
          className="h-8 min-w-0 bg-transparent text-sm outline-none sm:w-[128px]"
          aria-label="开始日期"
        />
        <span className="hidden text-xs text-stone-400 sm:inline">至</span>
        <input
          type="date"
          value={endDate}
          min={startDate || undefined}
          onChange={(event) => onChange(startDate, event.target.value)}
          className="h-8 min-w-0 bg-transparent text-sm outline-none sm:w-[128px]"
          aria-label="结束日期"
        />
      </div>
    </Field>
  );
}
